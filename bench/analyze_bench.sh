#!/bin/bash

# 分析bench.py的共享内存使用情况
ncu --set full \
    --metrics "local_load_transactions_per_request,\
local_store_transactions_per_request,\
shared_load_transactions_per_request,\
shared_store_transactions_per_request,\
shared_efficiency" \
    -o bench_shmem_analysis \
    python3 bench.py

# 可选：生成更详细的共享内存分析报告
# ncu --target-processes all \
#     --set full \
#     --metrics "all" \
#     -o bench_full_analysis \
#     python3 bench.py

echo "分析完成！结果文件：bench_shmem_analysis.ncu-rep"
