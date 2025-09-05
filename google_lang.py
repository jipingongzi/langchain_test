from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def merge_related_docs(raw_docs):
    """合并原始文档中同页码、同类型的关联元素（如标题+正文），避免拆分过细"""
    merged_docs = []
    current_doc = None  # 累积当前合并的文档对象

    for doc in raw_docs:
        page_num = doc.metadata.get("page_number", "未知")
        elem_type = doc.metadata.get("element_type", "text")

        if current_doc is None:
            current_doc = doc.copy()
            continue

        current_page = current_doc.metadata.get("page_number", "未知")
        current_type = current_doc.metadata.get("element_type", "text")

        if (page_num == current_page) and (elem_type == "text") and (current_type == "text"):
            current_doc.page_content += "\n" + doc.page_content
        else:
            merged_docs.append(current_doc)
            current_doc = doc.copy()

    if current_doc is not None:
        merged_docs.append(current_doc)

    return merged_docs


client = PersistentClient(path="./bms_chroma_db")
collection_name = "bms_system_docs"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

collection = client.get_or_create_collection(name=collection_name)
all_ids = collection.get()["ids"]

if all_ids:
    print(f"✅ 加载已存在的向量库集合: {collection_name}")
    print(f"   集合包含 {len(all_ids)} 个文本块")
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
else:
    print(f"🔧 创建新的向量库集合: {collection_name}")
    loader = UnstructuredLoader(
        api_key="iyLmzWtLoVU32XOlivzoBc6aByye8K",
        file_path="./TDD EPAM BMS system.pdf",  # 确保路径正确
        strategy="hi_res",  # 高精度解析（保留表格/图片标记）
        partition_via_api=True,
        coordinates=True,
    )
    docs = []
    try:
        print("📄 正在加载PDF文档内容...")
        for doc in loader.lazy_load():
            cleaned_metadata = {
                "page_number": doc.metadata.get("page_number", "未知"),
                "element_type": doc.metadata.get("element_type", "text"),
                "element_id": doc.metadata.get("element_id", "未知")
            }
            if cleaned_metadata["element_type"] in ["Image", "Table", "Figure"]:
                doc.page_content = f"【{cleaned_metadata['element_type']}内容】：{doc.page_content.strip()}"
            cleaned_doc = doc.copy()
            cleaned_doc.metadata = cleaned_metadata
            docs.append(cleaned_doc)

        print(f"   原始文档加载完成，共包含 {len(docs)} 个独立元素")
        merged_docs = merge_related_docs(docs)
        print(f"   关联元素合并完成，共包含 {len(merged_docs)} 个合并后元素")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", ". ", "? ", "! ", "; ", "\n", " "]
        )
        split_docs = text_splitter.split_documents(merged_docs)
        print(f"   文本分割完成，共生成 {len(split_docs)} 个初始文本块")

        min_chunk_length = 20
        filtered_docs = [doc for doc in split_docs if len(doc.page_content) >= min_chunk_length]
        print(
            f"   短分块过滤完成：过滤 {len(split_docs) - len(filtered_docs)} 个短块，剩余 {len(filtered_docs)} 个有效文本块")

        print("\n📊 前3个有效分块质量验证：")
        for i, doc in enumerate(filtered_docs[:3]):
            print(f"\n   分块{i + 1}：")
            print(f"   - 页码：第{doc.metadata['page_number']}页")
            print(f"   - 长度：{len(doc.page_content)} 字符")
            print(f"   - 内容预览：{doc.page_content[:150]}..." if len(
                doc.page_content) > 150 else f"   - 内容：{doc.page_content}")

        vectorstore = Chroma.from_documents(
            documents=filtered_docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )
        print(f"\n✅ 向量库创建完成，已写入 {len(filtered_docs)} 个有效文本块")

    except Exception as e:
        print(f"❌ 文档处理失败：{str(e)}")
        exit(1)

# -------------------------- 10. 初始化LLM（DeepSeek） --------------------------
llm = ChatOpenAI(
    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    streaming=False,
    temperature=0.3  # 适度提高灵活性，避免回答过于僵硬
)

# -------------------------- 11. 优化提示词模板 --------------------------
prompt_template = """你是技术文档问答专家，严格按以下规则回答：

1. 仅使用提供的上下文信息，不添加任何外部知识；
2. 优先引用标记【Image内容】【Table内容】【Figure内容】的信息，所有要点必须标注来源页码（如“来源：第3页”）；
3. 若上下文包含问题相关信息（即使不完整），需分点总结；若完全无相关信息，才回复“未找到相关信息”；
4. 用简洁的英文回答（因上下文为英文文档），避免中英文混杂。

上下文:
{context}

问题: {question}
回答:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------------- 12. 初始化检索问答链 --------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 适合短上下文（已优化分块长度，无需其他类型）
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k": 6
        }
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True  # 返回检索到的源文档，方便调试
)

# -------------------------- 13. 测试问答效果 --------------------------
test_questions = [
    "what is current problem?",
    "How we fix current problem?",
    "What result can we get?",
    "What is suppliers portal?"
]

print("\n" + "=" * 60)
print("📝 开始测试问答效果")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n【问题 {i}】: {question}")
    print("-" * 40)

    try:
        result = qa_chain.invoke({"query": question})

        print("🔍 检索到的相关文档：")
        for j, doc in enumerate(result["source_documents"], 1):
            print(
                f"   文档{j}：第{doc.metadata['page_number']}页 | {len(doc.page_content)}字符 | 预览：{doc.page_content[:100]}...")

        print("\n💡 LLM回答：")
        print(result["result"])

    except Exception as e:
        print(f"❌ 问答过程出错：{str(e)}")

    print("\n" + "-" * 60)