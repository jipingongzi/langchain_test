import warnings
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
import sys

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")

# 角色关系数据库 - 《白夜行》主要角色关系
CHARACTER_RELATIONS = {
    "桐原亮司": {
        "父亲": "桐原洋介",
        "母亲": "桐原弥生子",
        "关联人物": ["西本雪穗", "友彦", "唐泽礼子"],
        "关系描述": "与西本雪穗有着深厚而复杂的关系，为她做了很多违法的事情"
    },
    "西本雪穗": {
        "母亲": "西本文代",
        "养母": "唐泽礼子",
        "关联人物": ["桐原亮司", "高宫诚", "筱冢一成"],
        "关系描述": "外表优雅美丽，内心复杂，与桐原亮司相互依存"
    },
    "高宫诚": {
        "关联人物": ["西本雪穗", "三泽千都留"],
        "关系描述": "西本雪穗的第一任丈夫，后来离婚"
    },
    "筱冢一成": {
        "关联人物": ["西本雪穗", "筱冢康晴"],
        "关系描述": "对雪穗持怀疑态度，是筱冢康晴的堂弟"
    },
    "桐原洋介": {
        "儿子": "桐原亮司",
        "妻子": "桐原弥生子",
        "关系描述": "被儿子桐原亮司杀害的当铺老板"
    }
}


# 角色关系查询工具
def query_character_relation(character_name):
    """查询《白夜行》中特定角色的关系信息"""
    character_name = character_name.strip()
    if not character_name:
        return "请提供具体的角色名称"

    # 尝试精确匹配
    if character_name in CHARACTER_RELATIONS:
        relations = CHARACTER_RELATIONS[character_name]
        result = [f"{character_name}的关系信息："]

        for key, value in relations.items():
            if isinstance(value, list):
                result.append(f"- {key}：{', '.join(value)}")
            else:
                result.append(f"- {key}：{value}")

        return "\n".join(result)

    # 尝试模糊匹配
    similar_chars = [name for name in CHARACTER_RELATIONS if character_name in name]
    if similar_chars:
        return f"未找到'{character_name}'，是否要查询：{', '.join(similar_chars)}"

    return f"未找到关于'{character_name}'的关系信息"


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

    # 6. 初始化对话记忆 - 关键修复：使用chat_history作为内存键
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 7. 创建工具
    tools = [
        Tool(
            name="CharacterRelationQuery",
            func=query_character_relation,
            description="用于查询《白夜行》中角色之间的关系。当用户问起角色之间的关系、家庭背景或人物关联时使用。"
        )
    ]

    # 8. 初始化Agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # 9. 构建带记忆和检索的链
    def format_docs(docs):
        return "\n\n".join([f"[相关片段{i + 1}]: {doc.page_content}" for i, doc in enumerate(docs)])

    def log_retrieved_docs(docs):
        print("\n【检索到的相关片段】:")
        for i, doc in enumerate(docs):
            print(f"片段{i + 1}: {doc.page_content[:150]}...")
        return docs

    # 构建检索链
    retrieval_chain = (
            RunnableParallel({
                "context": retriever | log_retrieved_docs | format_docs,
                "question": RunnablePassthrough()
            })
            | ChatPromptTemplate.from_template("""
        基于以下上下文信息回答问题：
        {context}

        问题：{question}

        回答：
        """)
            | llm
            | StrOutputParser()
    )

    # 10. 交互循环
    print("\n问答助手就绪，输入 'exit' 退出。")
    print("注意：现在可以查询角色关系了，例如'桐原亮司和雪穗是什么关系？'")
    while True:
        try:
            question = input("\n请输入你的问题: ")
            if question.lower() == 'exit':
                print("再见！")
                break

            # 判断是否需要调用工具
            if any(character in question for character in CHARACTER_RELATIONS.keys()):
                print("回答: ", end="", flush=True)
                # 调用Agent处理角色关系查询
                response = agent.run(question)
                print(response)
            else:
                # 使用普通检索回答
                print("回答: ", end="", flush=True)
                full_response = []
                for chunk in retrieval_chain.stream(question):
                    full_response.append(chunk)
                    print(chunk, end="", flush=True)
                print()

                # 保存对话到记忆
                response_text = "".join(full_response)
                memory.save_context({"input": question}, {"output": response_text})

        except Exception as e:
            print(f"\n交互错误: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
