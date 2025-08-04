from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # 兼容DeepSeek
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings  # 使用本地嵌入模型
import sys


# 读取文档内容
def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文档时出错: {e}")
        return None


# 初始化DeepSeek LLM，启用流式传输
llm = ChatOpenAI(
    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    streaming=True  # 启用流式返回
)

# 初始化本地嵌入模型 - 不依赖API
# 这将下载并使用sentence-transformers的模型，第一次运行会较慢
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # 轻量级但效果好的模型
    model_kwargs={'device': 'cpu'},  # 使用CPU运行，避免GPU问题
    encode_kwargs={'normalize_embeddings': True}
)

# 加载并处理文档
document = load_document("docs.txt")
if not document:
    print("无法加载文档，程序退出")
    sys.exit(1)

# 分割文档为片段
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " "]
)
texts = text_splitter.split_text(document)

# 创建向量存储
vector_store = InMemoryVectorStore.from_texts(texts, embeddings)

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 检索最相关的3个片段

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""
你是一个关于《白夜行》的专家问答助手。请基于以下提供的上下文信息和历史对话回答用户的问题。
如果上下文信息不足以回答问题，请说明无法回答，不要编造信息。

上下文:
{context}

历史对话:
{history}

用户问题:
{question}

回答:
""")

# 初始化对话记忆
memory = ConversationBufferMemory(memory_key="history", return_messages=True)


# 格式化检索到的文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 构建RAG链
rag_chain = (
        {
            "context": retriever | format_docs,
            "history": RunnablePassthrough.assign(history=lambda x: memory.load_memory_variables({})["history"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)


# 流式回答函数
def stream_answer(question):
    full_response = []
    # 流式返回每个片段
    for chunk in rag_chain.stream(question):
        full_response.append(chunk)
        yield chunk
    # 将完整回答存入记忆
    memory.save_context({"input": question}, {"output": "".join(full_response)})


# 交互主函数
def main():
    print("《白夜行》问答助手已启动，输入 'exit' 退出程序")
    while True:
        try:
            question = input("\n请输入你的问题: ")
            if question.lower() == 'exit':
                print("感谢使用，再见！")
                break
            print("回答: ", end="", flush=True)
            # 流式输出回答
            for chunk in stream_answer(question):
                print(chunk, end="", flush=True)
            print()  # 换行
        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


if __name__ == "__main__":
    main()
