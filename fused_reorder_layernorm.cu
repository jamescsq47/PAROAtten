#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/AccumulateType.h>  // for acc_type<…>
#include <ATen/Dispatch.h>        // for AT_DISPATCH_… macros

using at::native::cuda_utils::BlockReduce;
using at::native::cuda_utils::BlockReduceSum;
using welford_t = at::native::WelfordData<float,int64_t>;
using ops_t     = at::native::WelfordOps<float,float,int64_t,thrust::pair<float,float>>;

constexpr int THREADS = 256;

// Helpers: map between (f,h,w) and flat index in both orders
__device__ __forceinline__
int64_t flat_FHW(int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    return f*(H*W) + h*W + w;
}
__device__ __forceinline__
int64_t flat_FWH(int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    return f*(W*H) + w*H + h;
}
__device__ __forceinline__
int64_t flat_WFH(int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    return w*(F*H) + f*H + h;
}
__device__ __forceinline__
int64_t flat_WHF(int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    return w*(H*F) + h*F + f;
}
__device__ __forceinline__
int64_t flat_HFW(int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    return h*(F*W) + f*W + w;
}
__device__ __forceinline__
int64_t flat_HWF(int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    return h*(W*F) + w*F + f;
}

// pattern_id: 0=FHW,1=FWH,2=WFH,3=WHF,4=HFW,5=HWF
__device__ int64_t to_flat(int pid, int64_t f, int64_t h, int64_t w, int64_t F, int64_t H, int64_t W) {
    switch(pid) {
        case 0: return flat_FHW(f,h,w,F,H,W);
        case 1: return flat_FWH(f,h,w,F,H,W);
        case 2: return flat_WFH(f,h,w,F,H,W);
        case 3: return flat_WHF(f,h,w,F,H,W);
        case 4: return flat_HFW(f,h,w,F,H,W);
        case 5: return flat_HWF(f,h,w,F,H,W);
    }
    return 0;
}

template <typename scalar_t, typename acc_t>
__global__ void fused_reorder_ln_kernel(
    const scalar_t* __restrict__ X, // [B,T,C]
    const acc_t*    __restrict__ shift, // [B,C]
    const acc_t*    __restrict__ scale, // [B,C]
    scalar_t*       __restrict__ Y,     // [B,T,C]
    int64_t B, int64_t F, int64_t H, int64_t W, int64_t C,
    acc_t eps,
    int pattern_id,
    int mode  // 0=reorder,1=inv_reorder
) {
    int b = blockIdx.x;
    int flat_idx = blockIdx.y;    // in [0,T)
    // decode flat_idx under one of the two domains:
    // if mode=0: flat_idx is "new index" under this pattern;
    // if mode=1: flat_idx is "orig index" under default FHW.
    int64_t f,h,w;
    int64_t idx_in,idx_out;
    if (mode==1) {
        // flat_idx = to_flat(pattern, f,h,w)
        // we need to invert it: brute-force by mapping f/h/w loops.
        // since F,H,W small, that's OK.
        int64_t found = -1;
        for (int64_t fi=0; fi<F && found<0; ++fi) {
            for (int64_t hi=0; hi<H && found<0; ++hi) {
                for (int64_t wi=0; wi<W; ++wi) {
                    if (to_flat(pattern_id,fi,hi,wi,F,H,W)==flat_idx) {
                        f=fi; h=hi; w=wi; found=1; break;
                    }
                }
            }
        }
        idx_in=flat_idx;
        idx_out=flat_FHW(f,h,w,F,H,W);
    } else if (mode==0){
        // reorder: flat_idx is orig = FHW
        int64_t t = flat_idx;
        f = t/(H*W);
        int64_t rem = t%(H*W);
        h = rem/W;
        w = rem%W;
        idx_in=flat_idx;
        idx_out=to_flat(pattern_id,f,h,w,F,H,W);
    }
    // original FHW-flat and new-flat:
    int64_t base_in  = (b*F*H*W + idx_in )*C;
    int64_t base_out = (b*F*H*W + idx_out)*C;

    // Welford on X[base_in:base_in+C]
    ops_t op{0,false};
    welford_t local{0,0,0,0};
    for (int i=threadIdx.x; i<C; i+=blockDim.x) {
        acc_t v = (acc_t)X[base_in+i];
        local = op.reduce(local,v,i);
    }
    extern __shared__ char smem[];
    welford_t* shared = (welford_t*)smem;
    welford_t zero{0,0,0,0};
    welford_t total = BlockReduce(local, op, zero, shared);
    float mean=0, m2=0;
    if(threadIdx.x==0) {
        thrust::tie(m2,mean) = op.project(total);
        //m2 /= (float)C;
        shared[0].mean   = mean;
        shared[0].m2 = m2;
    }
    __syncthreads();
    mean = shared[0].mean;
    m2   = shared[0].m2;
    acc_t invstd = rsqrt(m2+eps);

    // emit Y
    for(int i=threadIdx.x;i<C;i+=blockDim.x){
        acc_t x = (acc_t)X[base_in+i];
        acc_t g = (acc_t)1 + scale[b*C+i];
        acc_t s = shift[b*C+i];
        acc_t y = ((x-mean)*invstd)*g + s;
        //test
        //y=x;
        //test end
        Y[base_out+i] = (scalar_t)y;
    }
}

std::vector<at::Tensor> fused_reorder_layernorm_cuda_forward(
    const at::Tensor& x,
    const at::Tensor& shift,
    const at::Tensor& scale,
    int64_t F, int64_t H, int64_t W,
    double eps,
    int pattern_id,
    int mode
) {
    auto B = x.size(0), T = x.size(1), C = x.size(2);
    TORCH_CHECK(T==F*H*W,"T must equal F*H*W");
    auto y = at::empty_like(x);
    dim3 blocks(B,T);
    dim3 threads(THREADS);
    size_t shared = THREADS * sizeof(welford_t);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x.scalar_type(),           // 依赖输入 tensor 的 dtype
    "fused_reorder_ln",        // kernel tag，改成更有意义的名字
    [&] {
      using acc_t = at::acc_type<scalar_t, true>;   // 明确声明累加用的类型
      fused_reorder_ln_kernel<scalar_t, acc_t><<<blocks,threads,shared>>>(
          x.data_ptr<scalar_t>(),
          shift.data_ptr<acc_t>(),
          scale.data_ptr<acc_t>(),
          y.data_ptr<scalar_t>(),
          B, F, H, W, C,
          static_cast<acc_t>(eps),
          pattern_id, mode
      );
    });

    cudaDeviceSynchronize();
    return {y};
}