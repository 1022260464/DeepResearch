# test_rag.py
from tools.rag_tool import local_knowledge_retriever

def run_test():
    print("\n" + "="*50)
    print("🚀 开始测试本地文档检索工具")
    print("="*50)

    # 这里的 query 请换成你刚放入 data 文件夹的文档里【真正包含的内容】
    # 比如文档里写了“2024年利润增长30%”，你就搜“2024年利润情况”
    test_query = "公司针对早期肺部微小结节的识别准确率现在是多少？用了什么算法降低成本？"

    print(f"\n🗣️ 模拟用户提问: {test_query}\n")

    # 直接调用你封装好的工具
    try:
        result = local_knowledge_retriever.invoke(test_query)
        print("✅ 工具执行成功！大模型将看到以下返回内容：\n")
        print(result)
    except Exception as e:
        print(f"❌ 工具执行失败，报错信息: {e}")

if __name__ == "__main__":
    run_test()