# 安装库: pip install thop
import torch
from thop import profile
from model import RawAudioSSLNet # 假设你的模型在这里

# 1. 实例化模型
model = RawAudioSSLNet()

# 2. 创建一个对应的 Dummy Input (根据你的 input_len 和 num_mics)
# 你的模型输入是 (Batch, Mics, Length)，例如 (1, 4, 2048)
input_tensor = torch.randn(1, 4, 2048)

# 3. 计算 MACs 和 Params
macs, params = profile(model, inputs=(input_tensor, ))

print(f"计算量 (MACs): {macs}")
print(f"参数量 (Params): {params}")
# 转换为 Giga (10^9) 单位
print(f"计算量: {macs / 1e9:.4f} GMACs (Gops)")