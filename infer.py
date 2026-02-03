#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RawAudioSSLNet 声源定位推理脚本

输入: 4个音频文件路径 (对应4个麦克风), 支持 PCM 和 WAV 格式
输出: 声源方位角 (0-360度)

使用示例:
    python infer.py mic0.wav mic1.wav mic2.wav mic3.wav --model ssl_model_reg_35.pth
    python infer.py mic0.pcm mic1.pcm mic2.pcm mic3.pcm --dtype int16
    python infer.py mic0.wav mic1.wav mic2.wav mic3.wav  # 使用默认模型
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import librosa
from model import RawAudioSSLNet


# ============================================================================
# 音频处理函数
# ============================================================================
def load_wav(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    加载WAV文件 (使用librosa, 与训练时保持一致)
    
    Args:
        file_path: WAV文件路径
        target_sr: 目标采样率 (默认16000)
    
    Returns:
        归一化后的float32音频数据, 范围[-1, 1]
    
    Note:
        使用librosa加载音频有以下优势:
        - 自动处理各种WAV格式 (8/16/24/32-bit, float等)
        - 使用高质量重采样 (基于resampy, 使用Kaiser窗口滤波)
        - 自动归一化到 [-1, 1]
        - 自动转为单声道
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"WAV文件不存在: {file_path}")
    
    # librosa.load 自动处理:
    # 1. 多种音频格式和位深度
    # 2. 重采样 (使用高质量的 resampy)
    # 3. 归一化到 [-1, 1]
    # 4. 多声道转单声道 (mono=True)
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    return audio.astype(np.float32)


def load_pcm(
    file_path: str,
    sample_rate: int = 16000,
    dtype: np.dtype = np.int16
) -> np.ndarray:
    """
    加载PCM文件
    
    Args:
        file_path: PCM文件路径
        sample_rate: 采样率 (仅用于提示信息)
        dtype: PCM数据类型 (默认int16)
    
    Returns:
        归一化后的float32音频数据, 范围[-1, 1]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PCM文件不存在: {file_path}")
    
    # 读取原始PCM数据
    raw_data = np.fromfile(file_path, dtype=dtype)
    
    # 归一化到 [-1, 1]
    if dtype == np.int16:
        audio = raw_data.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        audio = raw_data.astype(np.float32) / 2147483648.0
    else:
        audio = raw_data.astype(np.float32)
    
    return audio


def load_audio(
    file_path: str,
    dtype: np.dtype = np.int16,
    target_sr: int = 16000
) -> np.ndarray:
    """
    自动检测文件类型并加载音频
    
    Args:
        file_path: 音频文件路径 (支持 .wav 和 .pcm)
        dtype: PCM文件的数据类型 (仅对PCM文件有效)
        target_sr: 目标采样率
    
    Returns:
        归一化后的float32音频数据, 范围[-1, 1]
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == '.wav':
        return load_wav(file_path, target_sr=target_sr)
    elif suffix == '.pcm':
        return load_pcm(file_path, sample_rate=target_sr, dtype=dtype)
    else:
        # 默认当作PCM处理
        print(f"[WARN] 未知文件格式 '{suffix}', 按PCM格式加载")
        return load_pcm(file_path, sample_rate=target_sr, dtype=dtype)


def preprocess_audio(
    audio_channels: list[np.ndarray],
    sample_length: int = 2048
) -> torch.Tensor:
    """
    预处理多通道音频数据
    
    Args:
        audio_channels: 4个通道的音频数据列表
        sample_length: 目标样本长度
    
    Returns:
        形状为 [1, 4, sample_length] 的tensor
    """
    # 找到最短的通道长度
    min_len = min(len(ch) for ch in audio_channels)
    
    if min_len < sample_length:
        raise ValueError(
            f"音频长度不足: 需要至少 {sample_length} 个样本, "
            f"但只有 {min_len} 个样本"
        )
    
    # 堆叠成多通道数组 [4, N]
    multi_ch = np.stack([ch[:min_len] for ch in audio_channels], axis=0)
    
    # 取中间的 sample_length 个样本 (避免首尾可能的静音)
    start = (min_len - sample_length) // 2
    cropped = multi_ch[:, start:start + sample_length]
    
    # 归一化 (保留通道间相对关系, 即ILD)
    max_amp = np.max(np.abs(cropped))
    if max_amp > 0:
        cropped = cropped / max_amp * 0.9
    
    # 转为tensor [1, 4, sample_length]
    tensor = torch.from_numpy(cropped.astype(np.float32)).unsqueeze(0)
    
    return tensor


def sincos_to_degree(sin_val: float, cos_val: float) -> float:
    """
    将 sin/cos 值转换为角度 (0-360度)
    
    Args:
        sin_val: 正弦值
        cos_val: 余弦值
    
    Returns:
        方位角 (0-360度)
    """
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    
    # 转换到 [0, 360) 区间
    if angle_deg < 0:
        angle_deg += 360.0
    
    return angle_deg


# ============================================================================
# 推理类
# ============================================================================
class SSLInference:
    """
    声源定位推理器
    
    Args:
        model_path: 模型权重文件路径
        device: 计算设备 ('cuda' 或 'cpu')
        sample_length: 输入样本长度
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        sample_length: int = 2048
    ):
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.sample_length = sample_length
        
        # 加载模型
        self.model = RawAudioSSLNet(num_mics=4, input_len=sample_length)
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] 模型已加载: {model_path}")
        print(f"[INFO] 使用设备: {self.device}")
    
    def _load_weights(self, model_path: str):
        """加载模型权重"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
    
    def infer_from_files(
        self,
        audio_paths: list[str],
        dtype: np.dtype = np.int16
    ) -> dict:
        """
        从音频文件进行推理 (支持 WAV 和 PCM 格式)
        
        Args:
            audio_paths: 4个音频文件路径列表 [mic0, mic1, mic2, mic3]
            dtype: PCM数据类型 (仅对PCM文件有效)
        
        Returns:
            包含推理结果的字典:
            {
                'angle_deg': 方位角(度),
                'sin': sin值,
                'cos': cos值
            }
        """
        if len(audio_paths) != 4:
            raise ValueError(f"需要4个音频文件, 但提供了 {len(audio_paths)} 个")
        
        # 加载4个通道的音频
        audio_channels = []
        for i, path in enumerate(audio_paths):
            audio = load_audio(path, dtype=dtype)
            audio_channels.append(audio)
            print(f"[INFO] 加载 mic{i}: {path} ({len(audio)} 样本)")
        
        # 预处理
        input_tensor = preprocess_audio(audio_channels, self.sample_length)
        input_tensor = input_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)  # [1, 2]
            sin_val = output[0, 0].item()
            cos_val = output[0, 1].item()
        
        # 转换为角度
        angle_deg = sincos_to_degree(sin_val, cos_val)
        
        return {
            'angle_deg': angle_deg,
            'sin': sin_val,
            'cos': cos_val
        }
    
    def infer_from_numpy(self, audio_channels: list[np.ndarray]) -> dict:
        """
        从numpy数组进行推理
        
        Args:
            audio_channels: 4个通道的音频数据列表, 每个为float32 [-1, 1]
        
        Returns:
            包含推理结果的字典
        """
        if len(audio_channels) != 4:
            raise ValueError(f"需要4个通道, 但提供了 {len(audio_channels)} 个")
        
        # 预处理
        input_tensor = preprocess_audio(audio_channels, self.sample_length)
        input_tensor = input_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            sin_val = output[0, 0].item()
            cos_val = output[0, 1].item()
        
        angle_deg = sincos_to_degree(sin_val, cos_val)
        
        return {
            'angle_deg': angle_deg,
            'sin': sin_val,
            'cos': cos_val
        }


# ============================================================================
# 主函数
# ============================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RawAudioSSLNet 声源定位推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python infer.py mic0.wav mic1.wav mic2.wav mic3.wav
  python infer.py mic0.pcm mic1.pcm mic2.pcm mic3.pcm --dtype int16
  python infer.py mic0.wav mic1.wav mic2.wav mic3.wav --model ssl_model_reg_40.pth
        """
    )
    
    parser.add_argument(
        'audio_files',
        nargs=4,
        metavar='AUDIO',
        help='4个音频文件路径 (支持 .wav 和 .pcm), 依次对应 mic0, mic1, mic2, mic3'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/mnt/chensenda/codes/sound/cnn/saved_2/ssl_model_reg_60.pth',
        help='模型权重文件路径 (默认: ssl_model_reg_35.pth)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='计算设备 (默认: 自动选择)'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='int16',
        choices=['int16', 'int32', 'float32'],
        help='PCM数据类型 (默认: int16)'
    )
    
    return parser.parse_args()


def main():
    """主入口函数"""
    args = parse_args()
    
    # 解析数据类型
    dtype_map = {
        'int16': np.int16,
        'int32': np.int32,
        'float32': np.float32
    }
    dtype = dtype_map[args.dtype]
    
    # 获取模型路径 (支持相对路径)
    model_path = args.model
    if not Path(model_path).is_absolute():
        # 如果是相对路径, 尝试在脚本所在目录查找
        script_dir = Path(__file__).parent
        model_in_script_dir = script_dir / model_path
        if model_in_script_dir.exists():
            model_path = str(model_in_script_dir)
    
    print("=" * 60)
    print("RawAudioSSLNet 声源定位推理")
    print("=" * 60)
    
    # 创建推理器
    inference = SSLInference(model_path=model_path, device=args.device)
    
    print("-" * 60)
    
    # 执行推理
    result = inference.infer_from_files(args.audio_files, dtype=dtype)
    
    print("-" * 60)
    print("[结果]")
    print(f"  声源方位角: {result['angle_deg']:.2f}°")
    print(f"  sin 值: {result['sin']:.4f}")
    print(f"  cos 值: {result['cos']:.4f}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
