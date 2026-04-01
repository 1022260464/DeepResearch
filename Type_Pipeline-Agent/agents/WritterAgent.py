import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .BaseDeepSeekModel import model


# 2. 定义写作系统指令 (WRITER_PROMPT)
WRITER_PROMPT = (
    "You are an expert technical writer and research synthesizer. "
    "Your task is to write a comprehensive, well-structured report based on the provided query and research context. "
    # 新增严厉的防缝合指令：
    "CRITICAL RULE: You must strictly distinguish between 'Internal Company Data' and 'Public Market Data'. "
    "NEVER attribute public achievements, external case studies, or competitors' deployment numbers to our company's specific product. "
    "Only use facts explicitly stated about our product for our product. "
    "Use the provided context to back up your claims. Do not make up facts."
)

# 构建 Prompt 模板
# 这里不仅需要原始的 query，还需要传入前面搜索到的上下文资料 (context)
writer_prompt = ChatPromptTemplate.from_messages([
    ('system', WRITER_PROMPT),
    ('human', "User Query: {query}\n\nResearch Context:\n{context}")
])

# 3. 定义结构化输出 (ReportData)
# 充分利用上一课的经验，给每个字段加上详细的 Field 描述
class ReportSection(BaseModel):
    heading: str = Field(description="The title or heading of this specific section. 章节标题。")
    content: str = Field(description="The detailed content/paragraphs for this section. 章节详细内容。")

class ReportData(BaseModel):
    title: str = Field(
        description="A clear and engaging title for the entire report. 整个报告的标题。"
    )
    executive_summary: str = Field(
        description="A brief 1-2 paragraph summary of the entire report's findings. 执行摘要/核心结论。"
    )
    sections: list[ReportSection] = Field(
        description="The main body sections of the report. 报告正文的各个章节。"
    )
    references: list[str] = Field(
        description="A list of sources or key points referenced from the provided context. 从上下文中提取的参考资料或关键来源列表。"
    )

# 4. 构建写作 Chain
writer_chain = writer_prompt | model.with_structured_output(ReportData)

# ==========================================
# 5. 核心改造：将执行逻辑和打印封装成函数
# ==========================================
def write_report(query: str, context: str) -> ReportData:
    """
    接收来自外层的 query 和 context，调用写作 Agent，并打印执行过程
    """
    print(f"\n[WritterAgent] ✍️ 正在根据搜索资料合成报告...")
    print(f"[WritterAgent] 用户问题：'{query}'")
    print(f"[WritterAgent] 上下文长度：{len(context)} 字符\n")

    try:
        # 执行 Chain
        final_report = writer_chain.invoke({
            "query": query,
            "context": context
        })

        # 格式化并打印结构化报告
        print(f"================ {final_report.title} ================\n")
        print(f"【执行摘要】\n{final_report.executive_summary}\n")

        for idx, sec in enumerate(final_report.sections, 1):
            print(f"【{idx}. {sec.heading}】\n{sec.content}\n")

        print("【参考来源】")
        for ref in final_report.references:
            print(f"- {ref}")

        return final_report

    except Exception as e:
        print(f"[WritterAgent] ❌ 运行出错：{e}")
        raise e

# 如果你想单独测试这个文件，可以保留下面的 main 判断
if __name__ == "__main__":
    test_query = "请问你对AI+教育有何看法"
    mock_context = """
    1. AI可以提供个性化学习体验，通过分析学生的学习进度自动调整题目难度（来源：教育科技前沿）。
    2. 教师可以使用AI自动批改作业和生成教案，从而减轻重复性工作负担，将更多精力放在师生互动上（来源：AI时代周刊）。
    3. 目前AI+教育面临数据隐私挑战，且过度依赖AI可能削弱学生的独立思考能力。此外，偏远地区缺乏设备导致数字鸿沟扩大（来源：全球教育报告2025）。
    """
    write_report(test_query, mock_context)