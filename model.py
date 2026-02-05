import torch
import torch.nn as nn
import torch.nn.functional as F

class RawAudioSSLNet(nn.Module):
    """
    基于原始波形的声源定位网络
    
    Args:
        num_mics: 麦克风数量 (默认4)
        input_len: 输入样本长度 (默认2048,16KHz,dt=2048/16000=0.128s)
    
    Output:
        [sin(angle), cos(angle)] 用于回归角度
    """
    def __init__(self, num_mics: int = 4, input_len: int = 2048):
        super(RawAudioSSLNet, self).__init__()
        
        C = 16
        # Stem: 第一层卷积 stride=1 捕捉微秒级相位差 消融表明这一步很重要
        self.stem = nn.Sequential(
            nn.Conv1d(num_mics, C, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True)
        ) # Bx4x2048 -> Bx16x2048

        # 骨干网络: 逐层下采样
        self.layer1 = self._make_layer(C, 2*C)    # Bx16x2048 -> Bx32x512
        self.layer2 = self._make_layer(2*C, 4*C)   # Bx32x512 -> Bx64x128
        self.layer3 = self._make_layer(4*C, 8*C)  # Bx64x64 ->  Bx128x64
        
        # 输出层: 预测 [sin, cos]
        self.predict = nn.Sequential(
            nn.Linear(8*C, 8*C),  # Fix missing comma here
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(8*C, 2)  # [sin_val, cos_val]
        )

    def _make_layer(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: BxDxT
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # BxDxT Global average pooling across the time axis
        x = x.mean(dim=-1)  # BxDxT -> BxD

        out = self.predict(x)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out


# class RawAudioSSLNet(nn.Module):
#     """
#     基于原始波形的声源定位网络
    
#     Args:
#         num_mics: 麦克风数量 (默认4)
#         input_len: 输入样本长度 (默认2048)
    
#     Output:
#         [sin(angle), cos(angle)] 用于回归角度
#     """
    
#     def __init__(self, num_mics: int = 4, input_len: int = 2048):
#         super(RawAudioSSLNet, self).__init__()
        
#         # Stem: 第一层卷积 stride=1 捕捉微秒级相位差
#         self.stem = nn.Sequential(
#             nn.Conv1d(num_mics, 64, kernel_size=15, stride=1, padding=7),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True)
#         )

#         # 骨干网络: 逐层下采样
#         self.layer1 = self._make_layer(64, 64)    # 2048 -> 1024
#         self.layer2 = self._make_layer(64, 128)   # 1024 -> 512
#         self.layer3 = self._make_layer(128, 256)  # 512 -> 256
#         self.layer4 = self._make_layer(256, 256)  # 256 -> 128
#         self.layer5 = self._make_layer(256, 512)  # 128 -> 64

#         # 全局平均池化
#         self.gap = nn.AdaptiveAvgPool1d(1)
        
#         # 输出层: 预测 [sin, cos]
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(256, 2)  # [sin_val, cos_val]
#         )

#     def _make_layer(self, in_c: int, out_c: int) -> nn.Sequential:
#         return nn.Sequential(
#             nn.Conv1d(in_c, out_c, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(out_c),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(out_c, out_c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(out_c),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.gap(x).flatten(1)
#         out = self.classifier(x)
#         out = torch.nn.functional.normalize(out, p=2, dim=1)
#         return out

# class DownBlock2xS2(nn.Module):
#     """A block with two stride=2 convolutions (total 4x downsampling), DV500-friendly: Conv + ReLU"""
#     def __init__(self, in_c: int, out_c: int, k: int = 5):
#         super().__init__()
#         p = k // 2
#         self.net = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=(1, k), stride=(1, 2), padding=(0, p)),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(out_c, out_c, kernel_size=(1, k), stride=(1, 2), padding=(0, p)),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class RawAudioSSLNet(nn.Module):
#     """
#     DV500 optimized version: two blocks (block1/block2), each with two stride=2 convolutions
#     Input:  (B, M, T)
#     Internal: (B, M, 1, T)
#     Output: (B, 2) -> [sin, cos]
#     """
#     def __init__(self, num_mics: int = 4, k_stem: int = 15, k: int = 5, input_len: int = 2048):
#         super().__init__()
#         C = 16
#         p_stem = k_stem // 2

#         # stem：建议 stride=1（保相位细节），下采样交给 block 内的两次 stride=2
#         self.stem = nn.Sequential(
#             nn.Conv2d(num_mics, C, kernel_size=(1, k_stem), stride=(1, 1), padding=(0, p_stem)),
#             nn.BatchNorm2d(C),
#             nn.ReLU(inplace=True),
#         )

#         # 两个 block：每个 block 内部两次 stride=2（总 /16）
#         self.block1 = DownBlock2xS2(C,   2*C, k=k)  # T -> T/4
#         self.block2 = DownBlock2xS2(2*C, 4*C, k=k)  # T -> T/16
#         # self.block3 = DownBlock2xS2(4*C, 8*C, k=k)  # T -> T/64
#         # GAP + 极简 head
#         # self.fc = nn.Linear(4*C, 2)

#         # 输出层: 预测 [sin, cos]
#         self.predict = nn.Sequential(
#             nn.Linear(4*C, 4*C),  # Fix missing comma here
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.2),
#             nn.Linear(4*C, 2)  # [sin_val, cos_val]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # (B,M,T) -> (B,M,1,T)
#         x = x.unsqueeze(2)

#         x = self.stem(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         # x = self.block3(x)
#         # (B,C,1,T') -> (B,C)
#         x = x.mean(dim=-1).squeeze(-1)
#         out = self.predict(x)
#         return F.normalize(out, p=2, dim=1)




# # ==========================================
# # 1. 粘贴或导入 Model 定义 (RawAudioSSLNet)
# # ==========================================
# class RawAudioSSLNet(nn.Module):
#     def __init__(self, num_mics=4, num_classes=360, input_len=2048):
#         # 🔥 注意: num_classes 改为 360，因为 Dataset 产生 0-359 的标签
#         # 🔥 注意: input_len 改为 2048，增加时间窗口长度
#         super(RawAudioSSLNet, self).__init__()
#         self.input_len = input_len
        
#         self.stem = nn.Sequential(
#             nn.Conv1d(num_mics, 32, kernel_size=64, stride=4, padding=30),
#             nn.BatchNorm1d(32), nn.ReLU(inplace=True)
#         )
#         self.layer1 = self._make_layer(32, 64)
#         self.layer2 = self._make_layer(64, 128)
#         self.layer3 = self._make_layer(128, 256)
#         self.layer4 = self._make_layer(256, 512)
        
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
#             nn.Linear(256, num_classes)
#         )

#     def _make_layer(self, in_c, out_c):
#         return nn.Sequential(
#             nn.Conv1d(in_c, out_c, 5, 2, 2), nn.BatchNorm1d(out_c), nn.ReLU(True),
#             nn.Conv1d(out_c, out_c, 3, 1, 1), nn.BatchNorm1d(out_c), nn.ReLU(True)
#         )

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.gap(x).flatten(1)
#         return self.classifier(x)