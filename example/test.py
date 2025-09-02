import torch

# 加载张量数据
tensor_path1 = '/home/chensiqi/PAROAtten/example/diff_of_kernel_and_dequantFA.pt'
data1 = torch.load(tensor_path1)
tensor_path2 = '/home/chensiqi/PAROAtten/example/diff_of_FA_and_dequantFA'
data2 = torch.load(tensor_path2)

print(f"原始张量形状: {data1.shape}")  # 应为 [1, 40, 75600, 128]
result1 = data1.mean(dim=(2, 3))
print(f"原始张量形状: {data2.shape}")  # 应为 [1, 40, 75600, 128]
result2 = data2.mean(dim=(2, 3))
print(result1)
print(result2)