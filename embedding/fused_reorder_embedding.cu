// fused_apply_rotary.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>


constexpr int THREADS = 256;

// FHW 解码
__device__ inline void decode_FHW(int64_t idx, int64_t H, int64_t W,
                                  int64_t &f, int64_t &h, int64_t &w) {
  f = idx / (H*W);
  int64_t rem = idx % (H*W);
  h = rem / W;
  w = rem % W;
}

// (f,h,w) -> flat 下各模式
__device__ inline int64_t to_flat(int pid,
    int64_t f, int64_t h, int64_t w,
    int64_t F, int64_t H, int64_t W) {
  switch(pid) {
    case 0: return f*(H*W) + h*W + w;              // FHW
    case 1: return f*(W*H) + w*H + h;              // FWH (self-inverse)
    case 2: return w*(F*H) + f*H + h;              // WFH ✔
    case 3: return w*(H*F) + h*F + f;              // WHF ✔
    case 4: return h*(F*W) + f*W + w;              // HFW ✔
    case 5: return h*(W*F) + w*F + f;              // HWF ✔
  }
  return 0;
}

template<typename scalar_t, typename acc_t>
__global__ void fused_apply_rotary_kernel(
    const scalar_t* __restrict__ x,    // [B, S, D]
    const scalar_t* __restrict__ cos,  // [S, D]
    const scalar_t* __restrict__ sin,  // [S, D]
    scalar_t*       __restrict__ y,    // [B, S, D]
    int64_t B, int64_t S, int64_t D,
    int64_t F, int64_t H, int64_t W,
    int pid,
    bool inv
) {
  int b = blockIdx.x;
  int s = blockIdx.y;
  int t = threadIdx.x;

  // 原序列 s → (f,h,w)
  int64_t f,h,w;
  decode_FHW(s, H, W, f,h,w);


  // 计算目标序号 sp
  int64_t sp=to_flat(pid, f,h,w, F,H,W);
  
  /*
  const scalar_t* x_ptr = x   + (b*S + s)*D;
  const scalar_t* c_ptr = cos + sp*D;
  const scalar_t* s_ptr = sin + sp*D;
  scalar_t*       y_ptr = y   + (b*S + s)*D;
  */
  // 对 x_ptr[0..D) 真正做 (real, imag) 交错旋转
    /*acc_t vx    = static_cast<acc_t>(x_ptr[i]);
    // x_rot = stack([-imag, real]) flatten，和 Python 一致：
    acc_t xrot  = ( (i & 1)==0 )
      ? -static_cast<acc_t>(x_ptr[i+1])
      :  static_cast<acc_t>(x_ptr[i-1]);
    */
    //scalar_t cc = c_ptr[i];
    //scalar_t ss = s_ptr[i];
    //acc_t vy = vx*cc + xrot*ss;
    //y_ptr[i] = static_cast<scalar_t>(vy);
  const scalar_t* x_ptr = x + (b * S + sp) * D;
  const scalar_t* cos_ptr = cos + s * D;
  const scalar_t* sin_ptr = sin + s * D;
  scalar_t*       y_ptr   = y + (b * S + sp) * D;

  // 4) 真正做 complex rotary：偶数维度负号交错
  for (int i = t; i < D; i += blockDim.x) {
    // 原始 real part
    acc_t vx = static_cast<acc_t>(x_ptr[i]);
    // 构造 x_rot：(-imag, real) 交错展开
    acc_t xrot = ((i & 1) == 0)
      ? -static_cast<acc_t>(x_ptr[i + 1])
      :  static_cast<acc_t>(x_ptr[i - 1]);

    // 取重排后的 cos/sin
    acc_t cc = static_cast<acc_t>(cos_ptr[i]);
    acc_t ss = static_cast<acc_t>(sin_ptr[i]);

    // 应用旋转
    acc_t vy = vx * cc + xrot * ss;
    y_ptr[i] = static_cast<scalar_t>(vy);
  }
}

std::vector<at::Tensor> fused_apply_rotary_cuda(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t F, int64_t H, int64_t W,
    const std::string& pattern,
    bool inv_reorder
) {
  auto B = x.size(0), S = x.size(1), D = x.size(2);
  TORCH_CHECK(S==F*H*W, "序列长度必须 F*H*W");
  static const std::vector<std::string> P = {
    "FHW","FWH","WFH","WHF","HFW","HWF"
  };
  int pid = -1;
  for(int i=0;i<6;i++){
    if (P[i]==pattern) { pid=i; break; }
  }
  TORCH_CHECK(pid>=0, "未知 pattern ", pattern);
  auto cos_s = cos.to(x.device(), x.scalar_type());
  auto sin_s = sin.to(x.device(), x.scalar_type());
  auto y = at::empty_like(x);
  dim3 blocks(B, S);
  dim3 threads(THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fuse_rotary", [&](){
    using acc_t = at::acc_type<scalar_t,true>;
    fused_apply_rotary_kernel<scalar_t,acc_t><<<blocks,threads>>>(
      x.data_ptr<scalar_t>(),
      cos_s.data_ptr<scalar_t>(),
      sin_s.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      B,S,D, F,H,W,
      pid, inv_reorder
    );
  });
  cudaDeviceSynchronize();
  return {y};
}
