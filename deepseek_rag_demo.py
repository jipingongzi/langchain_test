import warnings
import traceback  # 用于打印详细错误堆栈
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys

# 抑制PyTorch的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")

def load_document(file_path):
    try:
        print(f"尝试加载文档: {file_path}")  # 增加日志
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"文档加载成功，长度: {len(content)} 字符")  # 确认文档内容非空
            return content
    except Exception as e:
        print(f"加载文档失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()  # 打印详细错误堆栈
        return None

def main():
    print("《白夜行》问答助手启动中...")

    # 1. 加载文档（最可能出错的步骤）
    document = load_document("docs.txt")
    if not document:
        print("文档加载失败，程序无法继续，退出")
        return  # 用return代替sys.exit，避免强制退出导致日志不完整

    # 2. 初始化嵌入模型（可能因模型下载失败出错）
    try:
        print("初始化嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # 测试嵌入模型是否可用（生成一个简单向量）
        test_embedding = embeddings.embed_query("测试")
        print(f"嵌入模型初始化成功，测试向量维度: {len(test_embedding)}")
    except Exception as e:
        print(f"嵌入模型初始化失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return

    # 3. 分割文档
    try:
        print("分割文档为片段...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        texts = text_splitter.split_text(document)
        print(f"文档分割完成，得到 {len(texts)} 个片段")
        if len(texts) == 0:
            print("文档分割后为空，无法继续")
            return
    except Exception as e:
        print(f"文档分割失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return

    # 4. 创建向量存储
    try:
        print("创建向量存储...")
        vector_store = InMemoryVectorStore.from_texts(texts, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("向量存储创建成功")
    except Exception as e:
        print(f"向量存储创建失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return

    # 5. 初始化LLM（可能因API密钥/网络出错）
    try:
        print("初始化DeepSeek LLM...")
        llm = ChatOpenAI(
            api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            streaming=True
        )
        # 测试LLM是否可用（发送简单请求）
        test_response = llm.invoke("你好")
        print(f"LLM初始化成功，测试响应: {test_response.content[:20]}...")
    except Exception as e:
        print(f"LLM初始化失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return

    # 6. 初始化对话记忆
    try:
        print("初始化对话记忆...")
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        print("对话记忆初始化成功")
    except Exception as e:
        print(f"对话记忆初始化失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return

    # 7. 构建提示模板和RAG链
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    try:
        print("构建RAG链...")
        rag_chain = (
            {
                "context": retriever | format_docs,
                "history": RunnablePassthrough() | (lambda _: memory.load_memory_variables({})["history"]),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG链构建成功")
    except Exception as e:
        print(f"RAG链构建失败: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return

    # 8. 进入交互循环
    print("\n问答助手已准备就绪，输入 'exit' 退出程序")
    while True:
        try:
            question = input("\n请输入你的问题: ")
            if question.lower() == 'exit':
                print("感谢使用，再见！")
                break
            print("回答: ", end="", flush=True)
            full_response = []
            for chunk in rag_chain.stream(question):
                full_response.append(chunk)
                print(chunk, end="", flush=True)
            print()
            memory.save_context({"input": question}, {"output": "".join(full_response)})
        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"\n交互过程出错: {str(e)}")
            print("错误堆栈:")
            traceback.print_exc()

if __name__ == "__main__":
    main()