# tools/reader_tool.py
import requests
from langchain_core.tools import tool

@tool
def jina_reader_tool(url: str) -> str:
    """
    【深度阅读工具】
    当你在网页搜索结果中看到了一个非常有价值的 URL，但摘要（Snippet）信息不足以回答问题时，
    请提取该 URL，并使用此工具获取该网页的完整正文内容。
    """
    print(f"    📖 [深度阅读触发] 正在抓取网页正文: {url}")

    # Jina Reader 的魔法 URL
    jina_url = f"https://r.jina.ai/{url}"

    headers = {
        # Jina Reader 默认免费可用。如果后续遇到请求频率限制，可以去官网申请免费 API Key 并取消下方注释
        # "Authorization": "Bearer YOUR_JINA_API_KEY",
        "X-Return-Format": "markdown" # 强制返回 Markdown 格式，大模型最喜欢的格式
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        response.raise_for_status()

        content = response.text

        # 截断保护：取前 10000 个字符，防止超大网页消耗过多 Token
        max_length = 10000
        if len(content) > max_length:
            return content[:max_length] + "\n\n...[由于网页过长，后续内容已截断]..."
        return content

    except Exception as e:
        return f"读取网页失败，请尝试依赖其他搜索结果。错误信息: {str(e)}"