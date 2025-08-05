import warnings
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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

    # 3. 分割文档
    try:
        print("分割文档为片段...")
        text_splitter = CharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            separator="-----------------------------"
        )
        texts = text_splitter.split_text(document)
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
            search_kwargs={"k": 2}
        )
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

    # 添加获取记忆的函数
    def get_memory(_):
        return memory.load_memory_variables({})["history"]

    # 7. 构建提示模板和RAG链 - 关键修改：调整优先级
    prompt = ChatPromptTemplate.from_template("""
    回答问题时，请严格遵循以下优先级规则：
    1. 如果历史对话中包含与当前问题相关的明确信息（尤其是用户纠正过的内容），必须优先使用这些信息，但同时也要参考上下文信息
    2. 在历史对话没有相关信息的情况下，完全使用上下文信息
    3. 如果都没有相关信息，直接说"无法回答"，不要添加任何外部知识

    上下文信息（仅当历史对话无相关内容时使用）:
    {context}

    历史对话（包含用户明确纠正的信息，优先级最高）:
    {history}

    用户当前问题:
    {question}

    回答:
    """)

    def format_docs(docs):
        return "\n\n".join([f"[相关片段{i + 1}]: {doc.page_content}" for i, doc in enumerate(docs)])

    def log_retrieved_docs(docs):
        print("\n【检索到的相关片段】:")
        for i, doc in enumerate(docs):
            print(f"片段{i + 1}: {doc.page_content[:150]}...")
        return docs

    # 8. 构建带记忆优先的RAG链
    rag_chain = (
            {
                "context": retriever | log_retrieved_docs | format_docs,
                "history": RunnableLambda(get_memory),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    # 9. 交互循环
    print("\n问答助手就绪，输入 'exit' 退出。")
    print("注意：现在用户纠正的信息会优先于知识库内容！")
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

            # 保存对话到记忆
            response_text = "".join(full_response)
            memory.save_context({"input": question}, {"output": response_text})
            print(f"【已保存对话到记忆】")

        except Exception as e:
            print(f"\n交互错误: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
