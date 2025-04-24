import re
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取nohup.out文件
with open('nohup.out', 'r') as f:
    lines = f.readlines()

# 2. 初始化张量矩阵
sparse = np.ones((1, 48, 279, 279), dtype=bool)

# 3. 定义正则表达式匹配模式
pattern = r'batch:(\d+), head:(\d+), row (\d+), column:(\d+)'

# 4. 遍历每一行，更新张量矩阵
for line in lines:
    match = re.search(pattern, line)
    if match:
        batch = int(match.group(1))
        head = int(match.group(2))
        row = int(match.group(3))
        col = int(match.group(4))
        # 确保索引在范围内
        if 0 <= batch < 1 and 0 <= head < 48 and 0 <= row < 279 and 0 <= col < 279:
            sparse[batch, head, row, col] = False

# 5. 可视化处理
# 创建一个大的画布
plt.figure(figsize=(20, 20))
# 计算子图的行数和列数
rows = 8
cols = 6

for i in range(48):
    plt.subplot(rows, cols, i+1)
    plt.imshow(sparse[0, i, :, :], cmap='gray')
    plt.title(f'Head {i}')
    plt.axis('off')

# 调整子图间距
plt.tight_layout()
output_path = "cuda.png"
plt.savefig(output_path)
print(f"Visualization saved to {output_path}")
