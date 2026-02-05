
import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def download_model(repo_id, base_dir="/mnt/models"):
    """
    从 HuggingFace 下载模型到 /home/chensenda/models/<repo_id> 目录
    支持断点续传
    """
    # 模型保存目录
    model_dir = os.path.join(base_dir, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    print(f"开始下载模型 {repo_id} 到 {model_dir} ...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # 复制文件而不是建符号链接
        resume_download=True           # 断点续传
    )

    print(f"✅ 模型 {repo_id} 已成功下载到 {model_dir}")

def download_dataset(repo_id, base_dir="/mnt/chensenda/codes/sound/cnn_audio_SSL/speech_data"):
    """
    从 HuggingFace 下载模型到 /home/chensenda/models/<repo_id> 目录
    支持断点续传
    """
    # 模型保存目录
    model_dir = os.path.join(base_dir, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    print(f"开始下载模型 {repo_id} 到 {model_dir} ...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset", 
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # 复制文件而不是建符号链接
        resume_download=True           # 断点续传
    )

    print(f"✅ 模型 {repo_id} 已成功下载到 {model_dir}")

if __name__ == "__main__":

    model_name = "AISHELL/AISHELL-1"
    download_dataset(model_name)
