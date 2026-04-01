
***

# 🚀 DeepResearch Pipeline Agent

基于 Agentic Workflow（智能体工作流）的深度研究与报告生成助手。本项目结合了本地知识库检索（RAG）与大模型并发搜索能力，能够根据用户输入的研究主题，自动进行任务拆解、多线程信息收集，并最终合成结构化的专业研究报告。

## ✨ 核心特性

* **🧠 混合检索架构 (RAG + Web Search)**
    * 内置本地知识库初始化，使用 `sentence-transformers/all-MiniLM-L6-v2` 生成高质量文本向量。
    * 结合全网实时搜索，弥补本地知识库的时效性盲区。
* **⚙️ 智能体工作流 (Agentic Pipeline)**
    * **任务规划 (Planner)**：将宏观的复杂研究课题，智能拆解为多个具体可执行的子搜索计划。
    * **并发执行 (Concurrent Search)**：多线程同时拉取子任务数据（例如：市场分析、临床验证、政策审批等），极大提升研究效率。
    * **深度撰写 (Writer)**：汇总所有检索到的碎片化信息，自动生成排版精良、逻辑清晰的 Markdown 深度长文报告。
* **⏸️ 人类干预 (Human-in-the-loop)**
    * 工作流在“规划”与“执行”之间支持人工介入。
    * 用户可以在终端实时审核 Agent 生成的搜索计划，确认无误后按 `[Enter]` 继续执行，或随时干预调整。

## 🛠️ 环境依赖

推荐使用 Python 3.9+ 环境。

```bash
# 克隆项目
https://github.com/1022260464/LangGraph_learnigDemo.git
cd Type_Pipeline-Agent

# 安装核心依赖 (示例，可以根据实际 requirements.txt 补充，或直接使用 pip install -r requirements.txt)
pip install -r requirements.txt
# LangGraph:
pip install langgraph msgpack
```

*注：国内网络环境下，建议在环境中配置 `HF_ENDPOINT=https://hf-mirror.com` 以加速 HuggingFace 模型的下载。*

## 🚀 快速开始

运行主程序入口：

```bash
python main.py
```

**交互流程示例：**
1. 系统初始化本地知识库模型。
2. 输入你的深度研究 Prompt（例如：“请总结目前全网AI医疗影像诊断的发展现状...”）。
3. Agent 自动生成 5 个维度的搜索计划。
4. **人工审核**：终端提示 `⏸️ 工作流已暂停，等待人工审核...`，按回车继续。
5. 系统启动并发搜索，拉取数据。
6. 自动在终端/本地输出完整的《研究分析报告》。

## 📂 项目结构 (示例)

```text
DeepResearch/Type_Pipeline-Agent/
├── main.py                 # 主程序入口
├── workflow/
│   └── graph.py            # LangGraph 工作流定义与状态管理
├── agents/
│   ├── __init__.py
│   ├── BaseDeepSeekModel.py    # 基础deepseek模型
│   ├── SearchAgent.py      # 并发搜索 Agent
│   └── TaskAgent.py       # 报告撰写 Agent
│   └── writterAgent.py     # 报告撰写 Agent
└── tools/
    └── rag_tool.py # 本地知识库与向量检索工具
    └── reader_tool.py # 抓取网页工具
```

## 🗺️ 流程图 (Roadmap / TODO)

- [x] **Agent 工作流搭建**：实现 Plan -> Search -> Write 的基础图结构。
- [x] **Human-in-the-loop**：实现规划阶段的终端人工审批挂起机制。
- [x] **并发优化**：实现多子任务的并行网络检索，缩短等待时间。
- [ ] **动态 RAG 接入**：**[当前研发重点]** 将代码中硬编码的检索数据替换为真实的本地文档加载器（Document Loaders），支持 PDF、Word 等格式的动态向量化与检索。

## 🤝 贡献指南

欢迎提交 Pull Request 或 Issue 探讨功能演进！

---

### 💡 提示：
当你的 RAG 模块（例如支持了 PDF 上传解析或动态连接 Chroma/FAISS 数据库）开发完成后，你可以直接把 **TODO** 里的第四项打上勾 `[x]`，并在 README 中补充一下如何导入本地文档的说明就可以了！