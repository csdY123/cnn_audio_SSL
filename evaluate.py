#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RawAudioSSLNet 数据集评估脚本

评估模型在 quadrilateral_mic_dataset 上的性能

使用示例:
    python evaluate.py /path/to/quadrilateral_mic_dataset --model ssl_model_reg_35.pth
    python evaluate.py /path/to/quadrilateral_mic_dataset --model ssl_model_reg_35.pth --save results.csv
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# 导入推理模块
from infer import SSLInference, load_wav, preprocess_audio, sincos_to_degree


def angular_error(pred: float, gt: float) -> float:
    """
    计算角度误差 (考虑环形特性, 0度和360度相邻)
    
    Args:
        pred: 预测角度 (0-360)
        gt: 真实角度 (0-360)
    
    Returns:
        角度误差 (0-180)
    """
    diff = abs(pred - gt)
    return min(diff, 360 - diff)


class DatasetEvaluator:
    """数据集评估器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        sample_length: int = 2048
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型权重路径
            device: 计算设备
            sample_length: 样本长度
        """
        self.inference = SSLInference(
            model_path=model_path,
            device=device,
            sample_length=sample_length
        )
        self.sample_length = sample_length
    
    def evaluate_sample(self, angle_dir: Path) -> dict:
        """
        评估单个样本
        
        Args:
            angle_dir: 角度目录路径 (如 angle_045)
        
        Returns:
            包含预测结果和GT的字典
        """
        # 解析GT角度 (从目录名)
        dir_name = angle_dir.name  # e.g., "angle_045"
        gt_angle = int(dir_name.split('_')[1])
        
        # 加载4个麦克风的WAV文件
        mic_files = [angle_dir / f"mic_{i}.wav" for i in range(4)]
        
        # 检查文件是否存在
        for f in mic_files:
            if not f.exists():
                raise FileNotFoundError(f"文件不存在: {f}")
        
        # 加载音频
        audio_channels = []
        for f in mic_files:
            audio = load_wav(str(f))
            audio_channels.append(audio)
        
        # 预处理
        input_tensor = preprocess_audio(audio_channels, self.sample_length)
        input_tensor = input_tensor.to(self.inference.device)
        
        # 推理
        with torch.no_grad():
            output = self.inference.model(input_tensor)
            sin_val = output[0, 0].item()
            cos_val = output[0, 1].item()
        
        pred_angle = sincos_to_degree(sin_val, cos_val)
        error = angular_error(pred_angle, gt_angle)
        
        return {
            'gt_angle': gt_angle,
            'pred_angle': pred_angle,
            'error': error,
            'sin': sin_val,
            'cos': cos_val
        }
    
    def evaluate_dataset(self, dataset_dir: str, verbose: bool = True) -> dict:
        """
        评估整个数据集
        
        Args:
            dataset_dir: 数据集根目录
            verbose: 是否显示进度条
        
        Returns:
            评估结果字典
        """
        dataset_path = Path(dataset_dir)
        
        # 查找所有 angle_xxx 目录
        angle_dirs = sorted([
            d for d in dataset_path.iterdir()
            if d.is_dir() and d.name.startswith('angle_')
        ])
        
        if len(angle_dirs) == 0:
            raise ValueError(f"在 {dataset_dir} 未找到 angle_xxx 目录")
        
        print(f"[INFO] 发现 {len(angle_dirs)} 个样本")
        
        # 评估每个样本
        results = []
        errors = []
        
        iterator = tqdm(angle_dirs, desc="评估中") if verbose else angle_dirs
        
        for angle_dir in iterator:
            try:
                result = self.evaluate_sample(angle_dir)
                results.append(result)
                errors.append(result['error'])
            except Exception as e:
                print(f"\n[WARN] 跳过 {angle_dir.name}: {e}")
        
        # 统计指标
        errors = np.array(errors)
        
        metrics = {
            'total_samples': len(results),
            'mae': float(np.mean(errors)),  # 平均绝对误差
            'median_error': float(np.median(errors)),  # 中位数误差
            'std': float(np.std(errors)),  # 标准差
            'max_error': float(np.max(errors)),  # 最大误差
            'min_error': float(np.min(errors)),  # 最小误差
            'rmse': float(np.sqrt(np.mean(errors ** 2))),  # 均方根误差
            
            # 准确率 (不同阈值)
            'acc_5deg': float(np.mean(errors <= 5) * 100),  # ±5度准确率
            'acc_10deg': float(np.mean(errors <= 10) * 100),  # ±10度准确率
            'acc_15deg': float(np.mean(errors <= 15) * 100),  # ±15度准确率
            'acc_30deg': float(np.mean(errors <= 30) * 100),  # ±30度准确率
            'acc_45deg': float(np.mean(errors <= 45) * 100),  # ±45度准确率
        }
        
        return {
            'metrics': metrics,
            'results': results
        }


def print_metrics(metrics: dict):
    """打印评估指标"""
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"  样本总数:     {metrics['total_samples']}")
    print("-" * 60)
    print("误差统计:")
    print(f"  平均误差 (MAE):  {metrics['mae']:.2f}°")
    print(f"  中位数误差:      {metrics['median_error']:.2f}°")
    print(f"  标准差:          {metrics['std']:.2f}°")
    print(f"  均方根误差 (RMSE): {metrics['rmse']:.2f}°")
    print(f"  最小误差:        {metrics['min_error']:.2f}°")
    print(f"  最大误差:        {metrics['max_error']:.2f}°")
    print("-" * 60)
    print("准确率 (误差 ≤ 阈值):")
    print(f"  ±5° 准确率:   {metrics['acc_5deg']:.2f}%")
    print(f"  ±10° 准确率:  {metrics['acc_10deg']:.2f}%")
    print(f"  ±15° 准确率:  {metrics['acc_15deg']:.2f}%")
    print(f"  ±30° 准确率:  {metrics['acc_30deg']:.2f}%")
    print(f"  ±45° 准确率:  {metrics['acc_45deg']:.2f}%")
    print("=" * 60)


def save_results(results: list, output_path: str):
    """保存详细结果到CSV"""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gt_angle', 'pred_angle', 'error', 'sin', 'cos'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[INFO] 结果已保存到: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RawAudioSSLNet 数据集评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py /path/to/quadrilateral_mic_dataset
  python evaluate.py /path/to/dataset --model ssl_model_reg_40.pth
  python evaluate.py /path/to/dataset --save results.csv
        """
    )
    
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/mnt/chensenda/codes/sound/cnn_audio_SSL/saved_middle_ddp/ssl_model_ddp_345.pth',
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
        '--save', '-s',
        type=str,
        default=None,
        help='保存详细结果的CSV文件路径'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式 (不显示进度条)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 处理模型路径
    model_path = args.model
    if not Path(model_path).is_absolute():
        script_dir = Path(__file__).parent
        model_in_script_dir = script_dir / model_path
        if model_in_script_dir.exists():
            model_path = str(model_in_script_dir)
    
    print("=" * 60)
    print("RawAudioSSLNet 数据集评估")
    print("=" * 60)
    print(f"[INFO] 数据集: {args.dataset_dir}")
    print(f"[INFO] 模型: {model_path}")
    
    # 创建评估器
    evaluator = DatasetEvaluator(
        model_path=model_path,
        device=args.device
    )
    
    # 评估数据集
    eval_result = evaluator.evaluate_dataset(
        args.dataset_dir,
        verbose=not args.quiet
    )
    
    # 打印结果
    print_metrics(eval_result['metrics'])
    
    # 保存详细结果
    if args.save:
        save_results(eval_result['results'], args.save)
    
    return eval_result


if __name__ == "__main__":
    main()

