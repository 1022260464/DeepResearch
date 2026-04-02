import os
import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

# # 强制设定 Hugging Face 的下载源为国内镜像站
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置路径
DATA_DIR = "./data"                  # 存放你的 PDF 和 Word 的文件夹
FAISS_INDEX_DIR = "./faiss_index"    # 向量数据库保存到本地的文件夹

print("🧠 正在初始化本地知识库...")

# 1. 从本地绝对路径加载模型，切断网络请求
local_model_path = r"E:\CodeWorkPlace\DeepResearch\models\all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

# 全局变量存储 retriever
retriever = None

def init_vector_store():
    global retriever

    # 【机制 1：如果本地已经存过向量库，直接秒速加载】
    if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
        print("📂 发现已存在的本地向量库，正在直接加载...")
        # 允许加载本地危险反序列化（FAISS 默认安全机制，需开启 allow_dangerous_deserialization）
        vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 表示每次检索最相关的 3 个片段
        print("✅ 知识库加载完毕！")
        return

    # 【机制 2：如果本地没有向量库，则扫描 data 文件夹构建】
    print(f"🔍 未发现本地向量库，开始扫描 {DATA_DIR} 目录下的文档...")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"⚠️ 警告：未找到 {DATA_DIR} 文件夹，已自动创建。请放入 PDF 或 Word 文档后重启程序。")
        return

    # 获取所有 pdf 和 docx 文件路径
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    word_files = glob.glob(os.path.join(DATA_DIR, "*.docx"))
    all_files = pdf_files + word_files

    if not all_files:
        print(f"⚠️ 警告：在 {DATA_DIR} 中没有找到任何 PDF 或 Word 文件。")
        return

    all_docs = []

    # 逐个加载文件
    for file_path in all_files:
        print(f"📄 正在读取: {file_path}")
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"❌ 读取文件 {file_path} 失败: {e}")

    if not all_docs:
        return

    # 文本切分：将长文档切成 500 字的小块，块与块之间重叠 50 字防止语意截断
    print("✂️ 正在切分文本块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"✅ 共切分为 {len(split_docs)} 个文本块。")

    # 构建向量数据库
    print("🧮 正在将文本块转化为语义向量并构建 FAISS 数据库 (这可能需要一小会儿)...")
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # 保存到本地，下次启动直接秒进！
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"💾 向量库已持久化保存至: {FAISS_INDEX_DIR}")

    # 生成检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ 知识库初始化彻底完成！")

# 启动时执行初始化
init_vector_store()

# ---------------------------------------------------------
# Agent 调用的 Tool 定义
# ---------------------------------------------------------
@tool
def local_knowledge_retriever(query: str) -> str:
    """用于检索本地硬盘中的真实 PDF 和 Word 文档。如果用户的搜索意图涉及内部文档、私有数据或特定报告，请优先使用此工具！"""

    if retriever is None:
        return "本地知识库尚未就绪或文件夹中没有文档，请检查 data 目录。"

    print(f"\n[Tool Execution] 正在本地知识库中检索: {query}")
    docs = retriever.invoke(query)

    if not docs:
        return "未在本地文档中检索到相关信息。"

    # 将检索到的文档列表，拼接成一段大模型能看懂的纯文本
    result_text = "以下是从本地文档库检索到的相关信息片段：\n\n"
    for i, doc in enumerate(docs):
        # 提取文件名，去掉冗长的绝对路径，方便大模型阅读
        file_name = os.path.basename(doc.metadata.get('source', '未知文件'))
        page_num = doc.metadata.get('page', '未知') # Word 可能没有页码，PDF 有

        result_text += f"【片段 {i+1}】来源: {file_name} (第 {page_num} 页)\n"
        result_text += f"内容: {doc.page_content}\n"
        result_text += "-" * 30 + "\n"

    return result_text