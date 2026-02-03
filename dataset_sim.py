# dataset_sim.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra
import librosa
import os
import glob
import random

class DynamicRoomSimulator(Dataset):
    def __init__(self, audio_source_dir, sample_length=2048, epoch_length=2000):
        self.sample_length = sample_length
        self.epoch_length = epoch_length
        self.fs = 16000
        
        # -------------------------------------------------------------
        # 1. 加载音频文件 (支持 .wav 和 .flac)
        # -------------------------------------------------------------
        self.source_files = []
        for ext in ['*.wav', '*.flac']:
            # 使用 case-insensitive 的方式查找更稳健，但在 Linux 下通常通过扩展名控制即可
            self.source_files.extend(glob.glob(os.path.join(audio_source_dir, f"**/{ext}"), recursive=True))
            
        if len(self.source_files) == 0:
            raise ValueError(f"在 {audio_source_dir} 没找到音频文件 (.wav/.flac)!")
        
        print(f"动态仿真器已加载: 发现 {len(self.source_files)} 个源音频文件")

        # Unitree Go2 麦克风阵列定义 (4个麦克风, 3D坐标)
        self.mic_positions_local = np.array([
            [ 0.1035,  0.0235, 0.0], # mic0
            [ 0.1035, -0.0235, 0.0], # mic1
            [-0.1035, -0.0235, 0.0], # mic2
            [-0.1035,  0.0235, 0.0], # mic3
        ]).T # 转置为 3x4

    def __len__(self):
        return self.epoch_length

    def _get_random_room_params(self):
        """随机生成房间参数"""
        room_dim = np.array([
            np.random.uniform(3.0, 8.0), np.random.uniform(3.0, 8.0), np.random.uniform(2.5, 3.5)
        ])
        rt60 = np.random.uniform(0.15, 0.6)
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            max_order = min(max_order, 5) 
        except:
            e_absorption, max_order = 0.3, 3
        return room_dim, e_absorption, max_order

    def _load_random_source_clip(self, duration_sec=1.0):
        wav_path = random.choice(self.source_files)
        # 获取时长 (librosa读取有点慢，为了效率，实际工程中可以缓存时长，但这里先这样用)
        full_duration = librosa.get_duration(path=wav_path)
        
        if full_duration <= duration_sec:
            offset = 0.0
        else:
            offset = np.random.uniform(0, full_duration - duration_sec)
        
        y, sr = librosa.load(wav_path, sr=self.fs, offset=offset, duration=duration_sec)
        
        # 简单的源信号归一化，防止源信号太小或太大
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def __getitem__(self, idx):
        # -----------------------------------------------------------
        # 1. 准备环境 (随机房间参数)
        # -----------------------------------------------------------
        room_dim, e_absorption, max_order = self._get_random_room_params()
        
        # 创建房间
        room = pra.ShoeBox(
            room_dim, 
            fs=self.fs, 
            materials=pra.Material(e_absorption), 
            max_order=max_order
        )

        # -----------------------------------------------------------
        # 2. 放置麦克风 (随机位置)
        # -----------------------------------------------------------
        # 保证阵列离墙至少 0.5米
        mic_center = np.array([
            np.random.uniform(0.5, room_dim[0] - 0.5),
            np.random.uniform(0.5, room_dim[1] - 0.5),
            np.random.uniform(0.5, room_dim[2] - 0.5) 
        ])
        
        current_mic_locs = self.mic_positions_local + mic_center.reshape(3, 1)
        room.add_microphone_array(current_mic_locs)

        # -----------------------------------------------------------
        # 3. 放置声源 (随机角度 + 扩大距离 + 高度扰动)
        # -----------------------------------------------------------
        angle_deg = np.random.randint(0, 360)
        angle_rad = np.deg2rad(angle_deg)
        
        # 🔥【改进1】扩大距离范围：0.5m (近场) 到 5.0m (远场)
        dist = np.random.uniform(0.5, 5.0)
        
        src_x = mic_center[0] + dist * np.cos(angle_rad)
        src_y = mic_center[1] + dist * np.sin(angle_rad)
        
        # 🔥【安全补丁 A】高度随机扰动 (Z-axis Jitter)
        src_z = mic_center[2] + np.random.uniform(-0.2, 1.5)
        
        # 防止出界 (Clip X, Y, Z)
        src_x = np.clip(src_x, 0.1, room_dim[0]-0.1)
        src_y = np.clip(src_y, 0.1, room_dim[1]-0.1)
        src_z = np.clip(src_z, 0.1, room_dim[2]-0.1)
        
        # 重新计算由于Clip导致的真实角度 (Azimuth)
        real_dx = src_x - mic_center[0]
        real_dy = src_y - mic_center[1]
        real_angle_rad = np.arctan2(real_dy, real_dx)
        if real_angle_rad < 0: real_angle_rad += 2*np.pi
        
        # 最终 Label (0-359)
        label_deg = int(np.degrees(real_angle_rad))
        
        # 读取随机片段 (0.5秒)
        source_signal = self._load_random_source_clip(duration_sec=0.5)
        room.add_source([src_x, src_y, src_z], signal=source_signal)

        # -----------------------------------------------------------
        # 4. 运行物理仿真
        # -----------------------------------------------------------
        room.simulate()
        simulated_audio = room.mic_array.signals # Shape: (4, N)

        # -----------------------------------------------------------
        # 🔥【改进2】在线注入噪声 (Noise Injection)
        # -----------------------------------------------------------
        # 随机信噪比 (SNR): 5dB (吵闹) - 30dB (安静)
        target_snr_db = np.random.uniform(5.0, 30.0)
        
        # 计算信号能量
        sig_power = np.mean(simulated_audio ** 2)
        if sig_power > 0:
            # 根据 SNR 计算噪声能量
            noise_power = sig_power / (10 ** (target_snr_db / 10))
            # 生成高斯白噪声
            noise = np.random.normal(0, np.sqrt(noise_power), simulated_audio.shape)
            # 叠加
            simulated_audio = simulated_audio + noise

        # -----------------------------------------------------------
        # 5. 后处理与随机裁剪 (含静音防御)
        # -----------------------------------------------------------
        # 归一化 (保留 ILD, 避免除零)
        max_amp = np.max(np.abs(simulated_audio))
        if max_amp > 0:
            simulated_audio = simulated_audio / max_amp * 0.9
        
        signal_len = simulated_audio.shape[1]
        
        if signal_len > self.sample_length:
            max_start = signal_len - self.sample_length
            
            # 🔥【安全补丁 B】静音切片防御 (Silent Crop Protection)
            # 尝试 3 次随机切片，确保切到的片段有足够的能量
            for _ in range(3):
                start = np.random.randint(0, max_start + 1)
                cropped = simulated_audio[:, start : start + self.sample_length]
                
                # 能量阈值检测 (1e-5 是经验值)
                if np.mean(cropped**2) > 1e-5:
                    break
            else:
                # 如果3次都失败(极罕见)，兜底方案：取正中间
                start = max_start // 2
                cropped = simulated_audio[:, start : start + self.sample_length]
                
        else:
            # 补零 (Padding)
            pad_width = self.sample_length - signal_len
            cropped = np.pad(simulated_audio, ((0,0), (0, pad_width)))

        # 转 Tensor
        audio_tensor = torch.from_numpy(cropped.astype(np.float32))
        
        # 返回角度 Label
        label_tensor = torch.tensor(label_deg, dtype=torch.long) 

        return audio_tensor, label_tensor