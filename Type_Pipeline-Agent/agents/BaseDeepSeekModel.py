import os
from dotenv import load_dotenv
# LangChain / LangGraph 核心依赖
from langchain_deepseek import ChatDeepSeek
# 加载环境变量
load_dotenv(override=True)
Deepseek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 初始化模型
model = ChatDeepSeek(
    model_name='deepseek-chat',
    api_key=Deepseek_API_KEY
)