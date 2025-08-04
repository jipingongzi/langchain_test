import warnings
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")

def load_document(file_path):
    try:
        print(f"加载文档: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"文档加载成功，总长度: {len(content)} 字符")
            # 打印文档前100字符，确认内容正确
            print(f"文档前100字符: {content[:100]}...")
            return content
    except Exception as e:
        print(f"加载文档失败: {str(e)}")
        traceback.print_exc()
        return None

def main():
    print("《白夜行》问答助手启动中...")

    # 1. 加载文档
    document = load_document("docs.txt")
    if not document:
        print("文档加载失败，退出")
        return

    # 2. 初始化嵌入模型
    try:
        print("初始化嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        test_embedding = embeddings.embed_query("测试")
        print(f"嵌入模型可用，测试向量维度: {len(test_embedding)}")
    except Exception as e:
        print(f"嵌入模型初始化失败: {str(e)}")
        traceback.print_exc()
        return

    # 3. 分割文档（优化分割逻辑）
    try:
        print("分割文档为片段...")
        text_splitter = CharacterTextSplitter(
            chunk_size=100,  # 减小chunk_size，提高检索精度
            chunk_overlap=10,
            separator="-----------------------------"  # 更细粒度的分割符
        )
        texts = text_splitter.split_text(document)
        print(texts)
        print(f"文档分割完成，得到 {len(texts)} 个片段")

    except Exception as e:
        print(f"文档分割失败: {str(e)}")
        traceback.print_exc()
        return

    # 4. 创建向量存储并验证
    try:
        print("创建向量存储...")
        vector_store = InMemoryVectorStore.from_texts(texts, embeddings)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2}  # 增加检索数量，提高命中率
        )
        # 测试检索功能（用文档中的关键词测试）
        print("向量存储创建成功")
    except Exception as e:
        print(f"向量存储创建失败: {str(e)}")
        traceback.print_exc()
        return

    # 5. 初始化LLM
    try:
        print("初始化DeepSeek LLM...")
        llm = ChatOpenAI(
            api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            streaming=True
        )
        test_response = llm.invoke("你好")
        print(f"LLM响应正常: {test_response.content[:20]}...")
    except Exception as e:
        print(f"LLM初始化失败: {str(e)}")
        traceback.print_exc()
        return

    # 6. 初始化对话记忆
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # 7. 构建提示模板和RAG链（增强上下文权重）
    prompt = ChatPromptTemplate.from_template("""
    你必须严格基于以下上下文信息回答问题，不要添加任何外部知识。如果上下文没有相关信息，直接说"无法回答"。

    上下文信息（必须优先使用）:
    {context}

    历史对话（仅作参考）:
    {history}

    用户问题:
    {question}

    回答:
    """)

    def format_docs(docs):
        # 格式化时保留片段序号，方便调试
        return "\n\n".join([f"[相关片段{i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])

    # 构建RAG链（添加检索日志）
    def log_retrieved_docs(docs):
        print("\n【检索到的相关片段】:")
        for i, doc in enumerate(docs):
            print(f"片段{i+1}: {doc.page_content[:150]}...")  # 打印检索到的内容
        return docs

    rag_chain = (
        {
            "context": retriever | log_retrieved_docs | format_docs,  # 增加检索日志
            "history": RunnablePassthrough() | (lambda _: memory.load_memory_variables({})["history"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 8. 交互循环
    print("\n问答助手就绪，输入 'exit' 退出。可测试问题：'唐泽雪穗的性格' 或 '桐原亮司做了什么'")
    while True:
        try:
            question = input("\n请输入你的问题: ")
            if question.lower() == 'exit':
                print("再见！")
                break
            print("回答: ", end="", flush=True)
            full_response = []
            for chunk in rag_chain.stream(question):
                full_response.append(chunk)
                print(chunk, end="", flush=True)
            print()
            memory.save_context({"input": question}, {"output": "".join(full_response)})
        except Exception as e:
            print(f"\n交互错误: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()