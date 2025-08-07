import warnings
import traceback
import streamlit as st
from streamlit_chat import message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage

# 页面配置
st.set_page_config(
    page_title="《白夜行》问答助手",
    page_icon="📚",
    layout="wide"
)

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")

# 角色关系数据库
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
    character_name = character_name.strip()
    if not character_name:
        return "请提供具体的角色名称"

    if character_name in CHARACTER_RELATIONS:
        relations = CHARACTER_RELATIONS[character_name]
        result = [f"{character_name}的关系信息："]
        for key, value in relations.items():
            if isinstance(value, list):
                result.append(f"- {key}：{', '.join(value)}")
            else:
                result.append(f"- {key}：{value}")
        return "\n".join(result)

    similar_chars = [name for name in CHARACTER_RELATIONS if character_name in name]
    if similar_chars:
        return f"未找到'{character_name}'，是否要查询：{', '.join(similar_chars)}"

    return f"未找到关于'{character_name}'的关系信息"


def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"加载文档失败: {str(e)}")
        return None


# 初始化对话记忆（确保全局唯一）
def init_memory():
    if 'memory' not in st.session_state:
        # 明确指定记忆键和输出键，确保与链兼容
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,  # 返回Message对象（而非字符串），便于LLM理解
            output_key="output"
        )
        st.session_state.memory.save_context(
            {"input": "初始化对话"},
            {"output": "记忆系统已启动"}
        )  # 预存一条记录，避免空记忆


# 获取格式化的对话历史（转为LLM可理解的Message格式）
def get_formatted_chat_history():
    if 'memory' not in st.session_state:
        init_memory()
    # 从记忆中加载原始消息（HumanMessage/AIMessage对象）
    memory_vars = st.session_state.memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])

    # 调试：在侧边栏显示当前记忆内容（可删除）
    with st.sidebar.expander("当前对话记忆（调试）"):
        st.write([f"{msg.type}: {msg.content}" for msg in chat_history])

    return chat_history


# 初始化应用
def init_app():
    init_memory()  # 优先初始化记忆

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.processing = False

        with st.spinner("初始化问答助手..."):
            # 1. 加载文档
            st.session_state.document = load_document("docs.txt")
            if not st.session_state.document:
                st.error("文档加载失败，应用无法启动")
                return False

            # 2. 初始化嵌入模型
            try:
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                st.error(f"嵌入模型初始化失败: {str(e)}")
                return False

            # 3. 分割文档
            try:
                text_splitter = CharacterTextSplitter(
                    chunk_size=100,
                    chunk_overlap=10,
                    separator="-----------------------------"
                )
                st.session_state.texts = text_splitter.split_text(st.session_state.document)
            except Exception as e:
                st.error(f"文档分割失败: {str(e)}")
                return False

            # 4. 创建向量存储
            try:
                st.session_state.vector_store = InMemoryVectorStore.from_texts(
                    st.session_state.texts, st.session_state.embeddings
                )
                st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
            except Exception as e:
                st.error(f"向量存储创建失败: {str(e)}")
                return False

            # 5. 初始化LLM
            try:
                st.session_state.llm = ChatOpenAI(
                    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-chat",
                    streaming=True
                )
            except Exception as e:
                st.error(f"LLM初始化失败: {str(e)}")
                return False

            # 6. 创建工具
            tools = [
                Tool(
                    name="CharacterRelationQuery",
                    func=query_character_relation,
                    description="查询《白夜行》角色关系、家庭背景或人物关联时使用"
                )
            ]

            # 7. 初始化Agent（使用带记忆的聊天模板）
            st.session_state.agent = initialize_agent(
                tools,
                st.session_state.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=st.session_state.memory,  # 共享全局记忆
                verbose=True,
                handle_parsing_errors=True,
                # 关键：使用LangChain默认的聊天记忆模板，确保记忆正确传入
                agent_kwargs={
                    "prompt": ChatPromptTemplate.from_messages([
                        ("system", "你是《白夜行》专家，根据对话历史和工具回答问题。"),
                        MessagesPlaceholder(variable_name="chat_history"),  # 记忆变量
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                    ])
                }
            )

            # 8. 构建带记忆的检索链（核心修复）
            def format_docs(docs):
                return "\n\n".join([f"[片段{i + 1}]: {doc.page_content}" for i, doc in enumerate(docs)])

            def log_retrieved_docs(docs):
                st.session_state.retrieved_docs = docs
                return docs

            # 检索链：同时传入对话历史和文档上下文
            st.session_state.retrieval_chain = (
                    RunnableParallel({
                        "context": st.session_state.retriever | log_retrieved_docs | format_docs,
                        "question": RunnablePassthrough(),
                        "chat_history": RunnableLambda(lambda _: get_formatted_chat_history())  # 传入格式化记忆
                    })
                    | ChatPromptTemplate.from_messages([
                ("system", "基于对话历史和上下文回答《白夜行》相关问题，确保参考历史对话。"),
                MessagesPlaceholder(variable_name="chat_history"),  # 对话历史
                ("human", "上下文：{context}\n问题：{question}")  # 结合上下文和当前问题
            ])
                    | st.session_state.llm
                    | StrOutputParser()
            )

            # 9. 初始化对话显示历史
            if 'messages' not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "你好！我是《白夜行》问答助手，可记住对话历史~"}
                ]

            st.session_state.initialized = True
            return True
    return True


# 处理用户输入（确保记忆保存）
def handle_user_input(user_input):
    if not user_input or st.session_state.processing:
        return

    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        with st.empty():
            thinking_placeholder = st.info("正在结合历史对话思考...")

            # 区分查询类型
            if any(character in user_input for character in CHARACTER_RELATIONS.keys()):
                # Agent查询（自动使用记忆）
                response = st.session_state.agent.run(user_input)
            else:
                # 检索链查询（手动传入记忆并保存）
                full_response = []
                for chunk in st.session_state.retrieval_chain.stream(user_input):
                    full_response.append(chunk)
                response = "".join(full_response)
                # 强制保存到记忆（关键步骤）
                st.session_state.memory.save_context(
                    {"input": user_input},  # 用户输入
                    {"output": response}  # 助手回应
                )

            thinking_placeholder.empty()

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 显示检索文档
        if 'retrieved_docs' in st.session_state and st.session_state.retrieved_docs:
            with st.expander("相关文档片段"):
                for i, doc in enumerate(st.session_state.retrieved_docs):
                    st.info(f"片段{i + 1}: {doc.page_content}")
            del st.session_state.retrieved_docs

    except Exception as e:
        error_msg = f"处理错误: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    finally:
        st.session_state.processing = False


# 主界面
def main():
    st.title("《白夜行》问答助手 📚")
    st.markdown("---")

    init_memory()
    if not init_app():
        return

    # 侧边栏
    with st.sidebar:
        st.header("角色快速查询")
        for character in CHARACTER_RELATIONS.keys():
            if st.button(character):
                handle_user_input(f"介绍一下{character}")

        st.markdown("---")
        if 'texts' in st.session_state:
            st.info(f"文档片段数: {len(st.session_state.texts)}")
        st.caption("提示：可连续提问，助手会记住上下文~")

    # 显示对话历史
    chat_container = st.container()
    with chat_container:
        if 'messages' in st.session_state:
            for i, msg in enumerate(st.session_state.messages):
                message(
                    msg["content"],
                    is_user=msg["role"] == "user",
                    key=f"msg_{i}"
                )

    # 用户输入
    input_container = st.container()
    with input_container:
        col1, col2 = st.columns([8, 1])
        with col1:
            user_input = st.text_input("请输入问题:", placeholder="例如：先问'桐原亮司是谁'，再问'他为什么杀人'")
        with col2:
            submit_btn = st.button("发送", use_container_width=True)

        if submit_btn and user_input:
            handle_user_input(user_input)
            st.rerun()


if __name__ == "__main__":
    main()