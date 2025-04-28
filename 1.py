import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.version.cuda)