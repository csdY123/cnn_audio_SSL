#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RawAudioSSLNet Continuous Audio Inference with Visualization

Input: 4 audio file paths (corresponding to 4 microphones), supports PCM and WAV formats
Output: Sound source azimuth angles over time (0-360 degrees) with visualization

Usage examples:
    python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav
    python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav --hop 0.1
    python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav --hop 0.05 --save result.png
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import librosa
from model import RawAudioSSLNet
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
        Infer from numpy arrays
        
        Args:
            audio_channels: List of 4 channel audio data, each as float32 [-1, 1]
        
        Returns:
            Dictionary containing inference results
        """
        if len(audio_channels) != 4:
            raise ValueError(f"Need 4 channels, but got {len(audio_channels)}")
        
        # Preprocess
        input_tensor = preprocess_audio(audio_channels, self.sample_length)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
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
    
    def infer_continuous(
        self,
        audio_paths: list[str],
        hop_seconds: float = 0.1,
        sample_rate: int = 16000,
        dtype: np.dtype = np.int16
    ) -> dict:
        """
        Perform continuous inference on audio files with sliding window
        
        Args:
            audio_paths: List of 4 audio file paths [mic0, mic1, mic2, mic3]
            hop_seconds: Time interval between inferences (in seconds)
            sample_rate: Audio sample rate
            dtype: PCM data type (only for PCM files)
        
        Returns:
            Dictionary containing:
            {
                'timestamps': List of timestamps (seconds),
                'angles': List of azimuth angles (degrees),
                'sin_values': List of sin values,
                'cos_values': List of cos values
            }
        """
        if len(audio_paths) != 4:
            raise ValueError(f"Need 4 audio files, but got {len(audio_paths)}")
        
        # Load all 4 channels
        audio_channels = []
        for i, path in enumerate(audio_paths):
            audio = load_audio(path, dtype=dtype, target_sr=sample_rate)
            audio_channels.append(audio)
            print(f"[INFO] Loaded mic{i}: {path} ({len(audio)} samples, {len(audio)/sample_rate:.2f}s)")
        
        # Find minimum length across all channels
        min_len = min(len(ch) for ch in audio_channels)
        audio_channels = [ch[:min_len] for ch in audio_channels]
        
        # Calculate hop size in samples
        hop_samples = int(hop_seconds * sample_rate)
        
        # Calculate number of windows
        num_windows = (min_len - self.sample_length) // hop_samples + 1
        
        if num_windows <= 0:
            raise ValueError(
                f"Audio too short: need at least {self.sample_length} samples, "
                f"but only got {min_len}"
            )
        
        print(f"[INFO] Audio duration: {min_len/sample_rate:.2f}s")
        print(f"[INFO] Window size: {self.sample_length} samples ({self.sample_length/sample_rate*1000:.1f}ms)")
        print(f"[INFO] Hop size: {hop_samples} samples ({hop_seconds*1000:.1f}ms)")
        print(f"[INFO] Total windows: {num_windows}")
        
        # Perform sliding window inference
        timestamps = []
        angles = []
        sin_values = []
        cos_values = []
        
        # Stack audio channels [4, N]
        multi_ch = np.stack(audio_channels, axis=0)
        
        for i in range(num_windows):
            start = i * hop_samples
            end = start + self.sample_length
            
            # Extract window
            window = multi_ch[:, start:end]
            
            # Normalize (preserve ILD)
            max_amp = np.max(np.abs(window))
            if max_amp > 0:
                window = window / max_amp * 0.9
            
            # Convert to tensor
            input_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                sin_val = output[0, 0].item()
                cos_val = output[0, 1].item()
            
            angle_deg = sincos_to_degree(sin_val, cos_val)
            
            # Calculate timestamp (center of window)
            timestamp = (start + self.sample_length / 2) / sample_rate
            
            timestamps.append(timestamp)
            angles.append(angle_deg)
            sin_values.append(sin_val)
            cos_values.append(cos_val)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == num_windows - 1:
                print(f"[INFO] Progress: {i+1}/{num_windows} ({(i+1)/num_windows*100:.1f}%)")
        
        return {
            'timestamps': timestamps,
            'angles': angles,
            'sin_values': sin_values,
            'cos_values': cos_values
        }


# ============================================================================
# Visualization Functions
# ============================================================================
def visualize_results(
    timestamps: list,
    angles: list,
    save_path: str = None,
    show_plot: bool = True,
    title: str = "Sound Source Localization Results"
):
    """
    Visualize continuous inference results with time series and polar plot
    
    Args:
        timestamps: List of timestamps (seconds)
        angles: List of azimuth angles (degrees)
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
        title: Plot title
    """
    # Set style
    plt.style.use('dark_background')
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#00FFAA')
    
    # Color map based on angle
    colors = plt.cm.hsv(np.array(angles) / 360.0)
    
    # -------------------------------------------------------------------------
    # Left subplot: Time series plot
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#1a1a2e')
    
    # Plot angle over time with color gradient
    for i in range(len(timestamps) - 1):
        ax1.plot(
            timestamps[i:i+2], angles[i:i+2],
            color=colors[i], linewidth=1.5, alpha=0.8
        )
    
    # Scatter points
    scatter = ax1.scatter(
        timestamps, angles,
        c=angles, cmap='hsv', s=20, alpha=0.9,
        edgecolors='white', linewidths=0.3, zorder=5
    )
    
    ax1.set_xlabel('Time (s)', fontsize=11, color='#AAAAAA')
    ax1.set_ylabel('Azimuth Angle (°)', fontsize=11, color='#AAAAAA')
    ax1.set_title('Angle vs Time', fontsize=12, color='#00DDFF')
    ax1.set_ylim(-10, 370)
    ax1.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax1.grid(True, alpha=0.3, linestyle='--', color='#444444')
    ax1.tick_params(colors='#888888')
    
    # Add horizontal reference lines
    for angle in [0, 90, 180, 270, 360]:
        ax1.axhline(y=angle, color='#555555', linestyle=':', alpha=0.5)
    
    # -------------------------------------------------------------------------
    # Right subplot: Polar plot (radar view)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_facecolor('#1a1a2e')
    
    # Convert degrees to radians (adjust for polar coordinate: 0° = North/Up)
    angles_rad = np.deg2rad(90 - np.array(angles))  # Rotate so 0° is at top
    
    # Create time-based alpha for trail effect
    num_points = len(timestamps)
    alphas = np.linspace(0.2, 1.0, num_points)
    
    # Plot points with trail effect
    for i in range(num_points):
        ax2.scatter(
            angles_rad[i], 1.0,
            c=[colors[i]], s=50 * alphas[i], alpha=alphas[i],
            edgecolors='white', linewidths=0.2
        )
    
    # Plot current position (last point) with emphasis
    if len(angles) > 0:
        current_angle_rad = angles_rad[-1]
        ax2.annotate(
            '', xy=(current_angle_rad, 1.0), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->', color='#FF5555',
                lw=2.5, mutation_scale=15
            )
        )
        ax2.scatter(
            current_angle_rad, 1.0,
            c='#FF5555', s=150, marker='o',
            edgecolors='white', linewidths=2, zorder=10
        )
        # Add angle label
        ax2.annotate(
            f'{angles[-1]:.1f}°',
            xy=(current_angle_rad, 1.15),
            fontsize=12, fontweight='bold', color='#FF5555',
            ha='center', va='center'
        )
    
    # Configure polar plot
    ax2.set_ylim(0, 1.3)
    ax2.set_yticks([])
    ax2.set_theta_zero_location('N')  # 0 degrees at top
    ax2.set_theta_direction(-1)  # Clockwise
    ax2.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
        fontsize=10, color='#AAAAAA'
    )
    ax2.set_title('Polar View (Current Direction)', fontsize=12, color='#00DDFF', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', color='#444444')
    
    # Add direction labels
    directions = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
    for name, deg in directions.items():
        rad = np.deg2rad(90 - deg)
        ax2.annotate(
            name, xy=(rad, 1.45),
            fontsize=11, fontweight='bold', color='#00FFAA',
            ha='center', va='center'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
        print(f"[INFO] Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def visualize_animated(
    timestamps: list,
    angles: list,
    interval_ms: int = 50,
    save_path: str = None
):
    """
    Create animated visualization of sound source localization
    
    Args:
        timestamps: List of timestamps (seconds)
        angles: List of azimuth angles (degrees)
        interval_ms: Animation interval in milliseconds
        save_path: Path to save the animation (optional, .gif or .mp4)
    """
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Sound Source Localization - Live', fontsize=14, fontweight='bold', color='#00FFAA')
    
    # Time series subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#1a1a2e')
    ax1.set_xlim(min(timestamps) - 0.1, max(timestamps) + 0.1)
    ax1.set_ylim(-10, 370)
    ax1.set_xlabel('Time (s)', fontsize=11, color='#AAAAAA')
    ax1.set_ylabel('Azimuth Angle (°)', fontsize=11, color='#AAAAAA')
    ax1.set_title('Angle vs Time', fontsize=12, color='#00DDFF')
    ax1.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax1.grid(True, alpha=0.3, linestyle='--', color='#444444')
    
    # Polar subplot
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_facecolor('#1a1a2e')
    ax2.set_ylim(0, 1.3)
    ax2.set_yticks([])
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
        fontsize=10, color='#AAAAAA'
    )
    ax2.set_title('Polar View', fontsize=12, color='#00DDFF', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', color='#444444')
    
    # Initialize plot elements
    line, = ax1.plot([], [], color='#00FFAA', linewidth=1.5, alpha=0.8)
    scatter_time = ax1.scatter([], [], c=[], cmap='hsv', s=30, vmin=0, vmax=360)
    scatter_polar = ax2.scatter([], [], c='#FF5555', s=100)
    arrow = None
    angle_text = ax2.text(0, 0, '', fontsize=12, fontweight='bold', color='#FF5555', ha='center')
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10, color='#AAAAAA')
    
    def init():
        line.set_data([], [])
        scatter_time.set_offsets(np.empty((0, 2)))
        scatter_polar.set_offsets(np.empty((0, 2)))
        angle_text.set_text('')
        time_text.set_text('')
        return line, scatter_time, scatter_polar, angle_text, time_text
    
    def animate(frame):
        nonlocal arrow
        
        # Get data up to current frame
        t_data = timestamps[:frame+1]
        a_data = angles[:frame+1]
        
        if len(t_data) == 0:
            return line, scatter_time, scatter_polar, angle_text, time_text
        
        # Update time series
        line.set_data(t_data, a_data)
        scatter_time.set_offsets(np.c_[t_data, a_data])
        scatter_time.set_array(np.array(a_data))
        
        # Update polar plot
        current_angle = a_data[-1]
        current_rad = np.deg2rad(90 - current_angle)
        scatter_polar.set_offsets([[current_rad, 1.0]])
        
        # Update arrow
        if arrow:
            arrow.remove()
        arrow = ax2.annotate(
            '', xy=(current_rad, 1.0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#FF5555', lw=2.5)
        )
        
        # Update text
        angle_text.set_position((current_rad, 1.2))
        angle_text.set_text(f'{current_angle:.1f}°')
        time_text.set_text(f'Time: {t_data[-1]:.2f}s')
        
        return line, scatter_time, scatter_polar, angle_text, time_text, arrow
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(timestamps), interval=interval_ms, blit=False
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval_ms)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000//interval_ms)
        print(f"[INFO] Animation saved to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


# ============================================================================
# Main Function
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='RawAudioSSLNet Continuous Audio Inference with Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav
  python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav --hop 0.05
  python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav --save result.png
  python infer_continue.py mic0.wav mic1.wav mic2.wav mic3.wav --animate --save result.gif
        """
    )
    
    parser.add_argument(
        'audio_files',
        nargs=4,
        metavar='AUDIO',
        help='4 audio file paths (.wav or .pcm), corresponding to mic0, mic1, mic2, mic3'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/mnt/chensenda/codes/sound/cnn/saved_2/ssl_model_reg_60_best.pth',
        help='Model weights file path'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Compute device (default: auto-select)'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='int16',
        choices=['int16', 'int32', 'float32'],
        help='PCM data type (default: int16)'
    )
    
    parser.add_argument(
        '--hop',
        type=float,
        default=0.1,
        help='Time interval between inferences in seconds (default: 0.1)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save visualization to file (supports .png, .jpg, .gif, .mp4)'
    )
    
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Show animated visualization instead of static plot'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (useful when only saving)'
    )
    
    return parser.parse_args()


def main():
    """Main entry function"""
    args = parse_args()
    
    # Parse data type
    dtype_map = {
        'int16': np.int16,
        'int32': np.int32,
        'float32': np.float32
    }
    dtype = dtype_map[args.dtype]
    
    # Get model path (support relative path)
    model_path = args.model
    if not Path(model_path).is_absolute():
        script_dir = Path(__file__).parent
        model_in_script_dir = script_dir / model_path
        if model_in_script_dir.exists():
            model_path = str(model_in_script_dir)
    
    print("=" * 60)
    print("RawAudioSSLNet Continuous Audio Inference")
    print("=" * 60)
    
    # Create inference engine
    inference = SSLInference(model_path=model_path, device=args.device)
    
    print("-" * 60)
    
    # Perform continuous inference
    result = inference.infer_continuous(
        args.audio_files,
        hop_seconds=args.hop,
        dtype=dtype
    )
    
    print("-" * 60)
    
    # Print statistics
    angles = result['angles']
    print("[Statistics]")
    print(f"  Total inferences: {len(angles)}")
    print(f"  Mean angle: {np.mean(angles):.2f}°")
    print(f"  Std deviation: {np.std(angles):.2f}°")
    print(f"  Min angle: {np.min(angles):.2f}°")
    print(f"  Max angle: {np.max(angles):.2f}°")
    print("=" * 60)
    
    # Visualization
    if args.animate or (args.save and args.save.endswith(('.gif', '.mp4'))):
        visualize_animated(
            result['timestamps'],
            result['angles'],
            save_path=args.save
        )
    else:
        visualize_results(
            result['timestamps'],
            result['angles'],
            save_path=args.save,
            show_plot=not args.no_show
        )
    
    return result


if __name__ == "__main__":
    main()
