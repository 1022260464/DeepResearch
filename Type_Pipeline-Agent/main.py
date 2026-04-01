# main.py
import asyncio
from workflow.graph import app


async def run_research():
    """
    【异步版本】执行深度研究工作流
    """
    # 1. 定义初始问题
    query = "请总结目前全网AI医疗影像诊断的发展现状，并重点说明我们公司自研的'AI-Scan'产品落地情况和准确率数据。"
    print(f"🚀 开始执行异步深度研究任务: {query}\n" + "="*40)

    initial_state = {"query": query}

    # 2. 执行 LangGraph 工作流（支持人在回路中断）
    try:
        # 使用异步流式执行以支持中断
        thread_config = {"configurable": {"thread_id": "research_session_1"}}
        final_state = None

        async for event in app.astream(initial_state, config=thread_config):
            # 检查是否是中断事件（在 searcher 节点前）
            if "__interrupt__" in event:
                print("\n" + "="*40)
                print("⏸️ 工作流已暂停，等待人工审核...")
                print("="*40)

                # 获取当前状态（包含规划好的搜索计划）
                current_state = await app.aget_state(thread_config)
                search_plan = current_state.values.get("search_plan", None)

                if search_plan:
                    print("\n📋 即将执行的搜索计划：")
                    # 兼容 dict 和对象两种形式
                    if isinstance(search_plan, dict):
                        searches = search_plan.get('searches', [])
                    else:
                        searches = search_plan.searches
                    for i, item in enumerate(searches, 1):
                        query = item.get('query', '') if isinstance(item, dict) else item.query
                        print(f"  {i}. {query}")

                # 询问用户是否继续
                print("\n选项：")
                print("  [Enter] 直接继续执行")
                print("  [q] 退出程序")
                user_input = input("\n请输入选项: ").strip().lower()

                if user_input == "q":
                    print("\n👋 已退出程序")
                    return

                # 用户确认后继续执行（传入 None 表示继续）
                print("\n▶️ 继续执行并发搜索...\n")
                async for resume_event in app.astream(None, config=thread_config):
                    final_state = resume_event
            else:
                final_state = event

        # 3. 结果处理与展示
        print("\n✅ 报告已生成！\n" + "="*40)

        # 从最终状态中提取报告
        if final_state:
            # 尝试从不同位置获取最终报告
            report = None
            if isinstance(final_state, dict):
                if "writer" in final_state:
                    report = final_state["writer"].get("final_report")
                elif "final_report" in final_state:
                    report = final_state["final_report"]

            if report:
                print(f"📑 标题: {report.title}")
                print(f"💡 摘要: {report.executive_summary}\n")

                for sec in report.sections:
                    print(f"【{sec.heading}】\n{sec.content}\n")
            else:
                print("⚠️ 未能从状态中提取报告")

    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    程序入口，使用 asyncio.run 启动异步任务
    """
    asyncio.run(run_research())


if __name__ == "__main__":
    main()