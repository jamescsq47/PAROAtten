import os 

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (9, 4096, 4096))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (9, 4096, 11008 * 2))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (9, 4096, 4096 * 3))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (9, 5120, 5120))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (9, 5120, 13824 * 2))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (9, 5120, 5120 * 3))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (12, 4096, 4096))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (12, 4096, 11008 * 2))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (12, 4096, 4096 * 3))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (12, 5120, 5120))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (12, 5120, 13824 * 2))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (12, 5120, 5120 * 3))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (15, 4096, 4096))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (15, 4096, 11008 * 2))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (15, 4096, 4096 * 3))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (15, 5120, 5120))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (15, 5120, 13824 * 2))

# os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (15, 5120, 5120 * 3))

for i in range(5):
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 4096, 4096))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 4096, 11008))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 4096, 12288))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 4096, 13696))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 4096, 16384))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 11008, 4096))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 13696, 4096))
    os.system('python3 flat_gemm_test.py --batch-size %d --input-dim %d --output-dim %d' % (2 ** i, 16384, 4096))
