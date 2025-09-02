#!/usr/bin/env python3
import subprocess
import os
import signal
import sys
import shutil

def main():
    # 设置输出文件名
    output_file = "bench_shared_memory.ncu-rep"
    
    # 获取虚拟环境路径（如果存在）
    venv_path = "/home/chensiqi/anaconda3/envs/check"  # 修改为您的虚拟环境路径
    python_path = f"{venv_path}/bin/python3" if os.path.exists(venv_path) else sys.executable
    
    # 获取 ncu 的绝对路径
    ncu_path = shutil.which("ncu") or "/usr/local/cuda-12.2/bin/ncu"
    
    # 构建 ncu 命令
    cmd = [
        ncu_path,
        "--set", "full",
        "--target-processes", "all",
        "--metrics", "shared_utilization,shared_load_transactions_per_request,shared_store_transactions_per_request,l1tex__data_bank_conflicts_pipe_lsu_mem_shared",
        "--export", output_file,
        "--force-overwrite",
        python_path, "bench.py"
    ]
    
    # 打印完整命令用于调试
    print("Executing command:")
    print(" ".join(cmd))
    
    # 执行分析命令
    print(f"Output will be saved to: {output_file}")
    process = subprocess.Popen(cmd)
    
    try:
        process.wait()
        print("Analysis completed successfully!")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted, sending SIGTERM to ncu...")
        process.send_signal(signal.SIGTERM)
        process.wait()

if __name__ == "__main__":
    main()