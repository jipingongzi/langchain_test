from pathlib import Path
from dotenv import load_dotenv
import os
import warnings
import shutil
from PIL import Image
import pytesseract
from easyocr import Reader
from transformers import BlipProcessor, BlipForConditionalGeneration
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate  # 新增导入
from typing import List, Tuple


class ImageProcessor:
    def __init__(self):
        self.tesseract_lang = "eng+chi_sim"
        self.easyocr_reader = Reader(['en', 'ch_sim'], gpu=False)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✅ 图片处理器初始化完成（OCR+语义描述）")

    def ocr_image(self, img_path: Path) -> str:
        try:
            img = Image.open(img_path)
            ocr_text = pytesseract.image_to_string(img, lang=self.tesseract_lang)
            print("-------------------:" + ocr_text)
            if ocr_text.strip():
                return f"【图片OCR文字】：{ocr_text.strip()}"
            else:
                result = self.easyocr_reader.readtext(str(img_path), detail=0)
                easyocr_text = "\n".join(result)
                return f"【图片OCR文字】：{easyocr_text.strip() if easyocr_text else '未识别到文字'}"
        except Exception as e:
            return f"【图片OCR错误】：{str(e)[:50]}"

    def generate_image_caption(self, img_path: Path) -> str:
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = self.blip_processor(img, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return f"【图片语义描述】：{caption}"
        except Exception as e:
            return f"【图片描述错误】：{str(e)[:50]}"

    def process_single_image(self, img_path: Path) -> str:
        if not img_path.exists():
            return "【图片不存在】"
        ocr_text = self.ocr_image(img_path)
        caption_text = self.generate_image_caption(img_path)
        return f"\n=== 图片信息（路径：{img_path.name}）===\n{caption_text}\n{ocr_text}\n"


def load_pdf_with_images(pdf_path: str, temp_img_dir: str = "./temp_pdf_images") -> List[Document]:
    pdf_path = Path(pdf_path)
    temp_img_dir = Path(temp_img_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path.resolve()}")

    if temp_img_dir.exists():
        shutil.rmtree(temp_img_dir)
    temp_img_dir.mkdir(exist_ok=True)
    print(f"📁 临时图片目录：{temp_img_dir.resolve()}")

    loader = UnstructuredPDFLoader(
        str(pdf_path),
        strategy="fast",
        extract_images_in_pdf=True,
        image_output_dir_path=str(temp_img_dir)
    )
    docs = loader.load()
    print(f"✅ 加载PDF完成（共{len(docs)}页），提取图片数量：{len(list(temp_img_dir.glob('*.png')))}")

    img_processor = ImageProcessor()
    merged_docs = []
    for page_idx, doc in enumerate(docs, 1):
        page_text = doc.page_content.strip() or "【本页无文字】"

        page_images = list(temp_img_dir.glob(f"page_{page_idx}_image_*.png"))
        if page_images:
            image_texts = [img_processor.process_single_image(img) for img in page_images]
            merged_text = f"=== PDF第{page_idx}页 ===\n{page_text}\n" + "\n".join(image_texts)
        else:
            merged_text = f"=== PDF第{page_idx}页 ===\n{page_text}"

        merged_doc = Document(
            page_content=merged_text,
            metadata={"page": page_idx, "source": str(pdf_path.name)}
        )
        merged_docs.append(merged_doc)

    # shutil.rmtree(temp_img_dir)
    print(f"✅ PDF图文合并完成（共{len(merged_docs)}页，图片已转为文本）")
    return merged_docs


class UniversalEmbeddingsAdapter(Embeddings):
    def __init__(self, chroma_embedding_fn):
        self.chroma_embedding_fn = chroma_embedding_fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.chroma_embedding_fn.embed_texts(texts)
        except AttributeError:
            return self.chroma_embedding_fn(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.chroma_embedding_fn.embed_texts([text])[0]
        except AttributeError:
            return self.chroma_embedding_fn([text])[0]


def init_deepseek_llm() -> ChatOpenAI:
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-9b5776bd68e045f7ae2171077134b2a4")
    if not deepseek_api_key.startswith("sk-"):
        raise ValueError("DeepSeek API密钥格式错误，应以'sk-'开头")

    return ChatOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3,
        timeout=60,
        max_retries=2
    )


def build_vector_store(merged_docs: List[Document]) -> Chroma:
    chroma_client = PersistentClient(path="./chroma_pdf_with_images_db")
    collection_name = "pdf_qa_with_images"

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    langchain_embeddings = UniversalEmbeddingsAdapter(embedding_fn)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(merged_docs)
    print(f"✂️  文本分割完成（含图片转文本），共{len(split_docs)}个文本块")

    collections = chroma_client.list_collections()
    collection_exists = any(col.name == collection_name for col in collections)

    if collection_exists:
        print(f"📦 加载已存在的向量库")
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=langchain_embeddings
        )
        collection = chroma_client.get_collection(name=collection_name)
        if collection.count() == 0:
            vector_store.add_documents(split_docs)
            print(f"✅ 已添加{len(split_docs)}个文本块到向量库")
    else:
        print(f"🚀 创建新向量库（含图文信息）")
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=langchain_embeddings,
            client=chroma_client,
            collection_name=collection_name
        )

    return vector_store


def build_qa_chain(vector_store: Chroma, llm: ChatOpenAI) -> RetrievalQA:
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    # 关键修复：将字符串转换为PromptTemplate实例
    prompt_template = """
    基于以下参考内容（包含PDF的文字和图片转文本信息）回答用户问题：
    {context}

    回答要求：
    1. 若参考内容中有“图片OCR文字”或“图片语义描述”，必须优先结合这些信息回答；
    2. 明确区分文字信息和图片信息（如“根据PDF第3页的图片描述...”）；
    3. 若参考内容无相关信息，直接回复“未找到相关信息”；
    4. 用自然语言，保持简洁准确。

    用户问题：{question}
    回答：
    """
    # 转换为PromptTemplate格式
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}  # 使用正确的模板对象
    )


def interactive_qa(qa_chain: RetrievalQA):
    print("\n=== PDF图文知识问答系统 ===")
    print("提示：输入 'exit' 或 '退出' 结束对话（支持问图片相关问题）")
    while True:
        user_question = input("\n请输入你的问题：")
        if user_question.lower() in ["exit", "退出"]:
            print("对话结束，再见！")
            break
        if not user_question.strip():
            print("请输入有效问题")
            continue

        try:
            result = qa_chain.invoke({"query": user_question})
            print("\n=== 回答 ===")
            print(result["result"])

            print("\n=== 参考来源（含图片转文本信息） ===")
            for i, doc in enumerate(result["source_documents"], 1):
                page_num = doc.metadata.get("page", "未知")
                source_preview = doc.page_content[:200].replace("\n", " ") + "..."
                print(f"{i}. PDF第{page_num}页：{source_preview}")
        except Exception as e:
            error_msg = str(e).lower()
            if "timed out" in error_msg:
                print("\n❌ 请求超时：检查网络或DeepSeek API状态")
            else:
                print(f"\n❌ 错误：{str(e)}")


if __name__ == "__main__":
    load_dotenv()
    pdf_path = "TDD EPAM BMS system.pdf"

    try:
        print("=== 初始化PDF图文问答系统 ===")
        llm = init_deepseek_llm()
        print("✅ DeepSeek LLM初始化完成")

        merged_docs = load_pdf_with_images(pdf_path)

        vector_store = build_vector_store(merged_docs)
        print("✅ 向量库构建完成（含图片转文本信息）")

        qa_chain = build_qa_chain(vector_store, llm)
        interactive_qa(qa_chain)

    except Exception as e:
        print(f"\n❌ 初始化失败：{str(e)}")
