import os
from huggingface_hub import snapshot_download

# 1. 强制在代码级别配置国内镜像源，绝对不会连不上！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 设置你要保存模型的本地绝对路径
local_path = r"/models/all-MiniLM-L6-v2"

print("🚀 开始从国内镜像源下载模型，请稍候...")

# 3. 启动下载
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=local_path,
    local_dir_use_symlinks=False  # 加上这个，防止 Windows 系统因为权限问题在创建软链接时报错
)

print(f"✅ 太棒了！模型已成功完整下载到: {local_path}")