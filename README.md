# PAROAttention (Hardware code)
This package includes PAROAttention codes with quantization and sparsity implementation. We build PAROAttention based on the [SageAttn V2](https://github.com/thu-ml/SageAttention) code, and further integrate the tailored designs for $PV^{T}$ INT8 quantization and block-wise sparsity. Additionally, this package also includes the RoPE kernel with fused permutation mentioned in our paper. 

## Installation
- `python >= 3.9`, `torch >= 2.3.0`, `CUDA >= 11.8`
- `flash-attn` for benchmarking
- compile from source
      `
            python setup.py install
      `
      or 
      `
            pip install -e .
      `
- Note that this version only supports **Ampere (sm80)** GPUs, such as A100, A800 etc.

## Benchmarking
- Attention acceleration under varying sparsity (0.2, 0.3, 0.5 density)

      cd bench
      python test.py

- Rope kernel with and without permutation

      cd bench
      python overhead.py

- Benchmark the baseline implementation such as [FlashAttention V2](https://github.com/Dao-AILab/flash-attention)

      cd bench
      python bench_baseline.py --method fa2


