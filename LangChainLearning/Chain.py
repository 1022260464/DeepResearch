import os
from dotenv import load_dotenv
# 导入 DeepSeek 模型和 Tavily 搜索工具
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
# 1. 加载 .env 文件中的环境变量
# override=True 确保 .env 文件中的变量会覆盖系统中可能存在的同名变量
load_dotenv(override=True)

# 2. 创建初始模型 (DeepSeek)
# 注意：因为环境变量中已经有了 DEEPSEEK_API_KEY，ChatDeepSeek 会自动读取它
# 你不需要再手动通过 os.getenv 传给 api_key 参数，保持代码清爽
model = ChatDeepSeek(
    model_name="deepseek-chat",
    temperature=0.7 # 你可以根据需要调整温度（创造力）
)
# 3. 顺手把你的 Tavily 搜索工具也初始化出来
search_tool = TavilySearch(max_results=3)
# ====================================================================
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt_template = ChatPromptTemplate([
    ("system", "你是一个乐意助人的助手，请根据用户的问题给出回答"),
    ("user", "这是用户的问题： {topic}， 请用 yes 或 no 来回答")
])

# 直接使用模型 + 输出解析器
bool_qa_chain = prompt_template | model | StrOutputParser()
# 测试
question = "请问 1 + 1 是否 大于 2？"
result = bool_qa_chain.invoke({'topic':question})
print(result)
