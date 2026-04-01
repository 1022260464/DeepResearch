# schema/state.py
from typing import TypedDict, List
import sys
import os

# 添加父目录到路径，以便导入 agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入你在 Agent 中定义的 Pydantic 模型
from agents.TaskAgent import WebSearchPlan
from agents.WritterAgent import ReportData

class ResearchState(TypedDict):
    query: str
    search_plan: WebSearchPlan
    search_results: List[str]
    final_report: ReportData