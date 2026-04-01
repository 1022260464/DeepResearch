# tools/rag_tool.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import  tool
from langchain_core.documents import Document

print("🧠 初始化本地知识库 (首次加载模型可能需要几秒)...")

# 1. 加载轻量级开源嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. 模拟本地机密文档 (真实场景下，你可以用 DocumentLoader 加载 PDF/TXT)
local_docs = [
    Document(
        page_content="【内部机密】2024年公司在AI+医疗影像领域的研发投入达到5000万元。我们的核心产品'AI-Scan'已在三甲医院落地，使得肺结节的早期筛查准确率提升至98%。",
        metadata={"source": "2024_AI医疗业务财报.pdf"}
    ),
    Document(
        page_content="【技术白皮书】我们的最新自研算法采用了多模态融合技术，不仅能看CT，还能结合病人的电子病历(EMR)进行综合诊断，预计2025年通过药监局审批。",
        metadata={"source": "多模态医疗AI技术白皮书_内部版.pdf"}
    )
]

# 3. 构建本地向量数据库
vector_store = FAISS.from_documents(local_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# # 4. 封装为 Agent 可调用的 Tool
# local_retriever_tool = create_retriever_tool(
#     retriever,
#     name="local_knowledge_retriever",
#     description="用于检索公司内部的私有文档、医疗财报、技术白皮书和机密数据。如果用户的搜索意图涉及公司内部进展、自研技术或内部数据，请务必优先使用此工具！"
# )
@tool
def local_knowledge_retriever(query: str) -> str:
    """用于检索公司内部的私有文档、医疗财报、技术白皮书和机密数据。如果用户的搜索意图涉及公司内部进展、自研技术或内部数据，请务必优先使用此工具！"""

    # 1. 执行向量库检索
    docs = retriever.invoke(query)

    # 2. 将检索到的文档列表，拼接成一段大模型能看懂的纯文本
    result_text = ""
    for i, doc in enumerate(docs):
        result_text += f"文档 {i+1} 来源: {doc.metadata.get('source')}\n"
        result_text += f"内容: {doc.page_content}\n\n"

    return result_text