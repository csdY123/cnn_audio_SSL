import torch
import soundfile as sf
import os
import numpy as np
from dataset_sim import DynamicRoomSimulator

def export_samples(output_dir="test_samples", num_samples=5):
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 实例化你的 Simulator (请确保路径正确)
    # 这里 sample_length 设长一点 (比如 2秒)，方便听出效果
    fs = 16000
    dataset = DynamicRoomSimulator(
        audio_source_dir="./speech_data/LibriSpeech/dev-clean", 
        sample_length=fs * 5,  # 采样2秒钟的数据
        epoch_length=num_samples
    )

    print(f"--- Start Generating {num_samples} Samples ---")

    for i in range(num_samples):
        # 获取一个样本
        # audio_tensor shape: [4, 32000] (4个麦克风, 2秒)
        # label_deg: 角度 (0-359)
        audio_tensor, label_deg = dataset[i]
        
        audio_np = audio_tensor.numpy()
        angle = label_deg.item()
        
        # 转换成双声道 WAV (取 mic0 和 mic1 分别放在左右声道)
        # 这样戴耳机能听到明显的相位差和声源方位
        stereo_output = audio_np[[3, 0], :].T 
        
        file_name = os.path.join(output_dir, f"sample_{i}_angle_{angle}.wav")
        sf.write(file_name, stereo_output, fs)
        
        print(f"Saved: {file_name} | True Angle: {angle}°")

if __name__ == "__main__":
    # 执行生成
    export_samples()