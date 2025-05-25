source /home/hongke/space/anaconda3/bin/activate
conda activate torch
cd /home/hongke/space/repo/Inficom/script/ops
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs1_256 --details-all python3 decode_attn_test.py --batch-size 1 --cache-len 256
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs1_2048 --details-all python3 decode_attn_test.py --batch-size 1 --cache-len 2048

/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs2_256 --details-all python3 decode_attn_test.py --batch-size 2 --cache-len 256
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs2_2048 --details-all python3 decode_attn_test.py --batch-size 2 --cache-len 2048

/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs4_256 --details-all python3 decode_attn_test.py --batch-size 4 --cache-len 256
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs4_2048 --details-all python3 decode_attn_test.py --batch-size 4 --cache-len 2048

/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs8_256 --details-all python3 decode_attn_test.py --batch-size 8 --cache-len 256
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs8_2048 --details-all python3 decode_attn_test.py --batch-size 8 --cache-len 2048

/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs16_256 --details-all python3 decode_attn_test.py --batch-size 16 --cache-len 256
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o attn_util_bs16_2048 --details-all python3 decode_attn_test.py --batch-size 16 --cache-len 2048
