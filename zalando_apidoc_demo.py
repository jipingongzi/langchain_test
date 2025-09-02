import yaml
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Optional, Any
import uuid

chroma_client = chromadb.PersistentClient(path="./chroma_openapi_db")  # 数据会存在当前目录的 chroma_openapi_db 文件夹


local_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 轻量级本地模型，无需网络
)

def get_or_create_collection(client: chromadb.Client, name: str, embedding_fn) -> chromadb.Collection:
    """安全获取或创建ChromaDB集合，避免重复创建错误"""
    try:
        # 先尝试获取集合
        collection = client.get_collection(name=name, embedding_function=embedding_fn)
        print(f"成功获取已存在的集合: {name}")
        return collection
    except ValueError:
        # 集合不存在时创建
        collection = client.create_collection(name=name, embedding_function=embedding_fn)
        print(f"成功创建新集合: {name}")
        return collection


collection = get_or_create_collection(
    client=chroma_client,
    name="openapi_specs",
    embedding_fn=local_embedding_fn
)

llm = ChatOpenAI(
    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",  # DeepSeek聊天模型正确名称（避免模型不存在错误）
    temperature=0.3,
    timeout=15  # 增加超时时间，避免网络波动导致失败
)

description_prompt = PromptTemplate(
    input_variables=["method", "path", "summary", "description"],
    template="""
    请为以下API生成一段简洁的自然语言描述（50-80字），包含：
    1. HTTP方法和路径 2. 核心功能 3. 用途
    不要添加额外信息，语言简洁专业。

    API信息：
    - 方法：{method}
    - 路径：{path}
    - 摘要：{summary}
    - 详细描述：{description}

    生成结果：
    """
)

description_chain = LLMChain(llm=llm, prompt=description_prompt)


def load_and_parse_openapi(yaml_path: str) -> Optional[Dict]:
    """加载并解析OpenAPI YAML，增加文件路径验证"""
    if not yaml_path.endswith((".yaml", ".yml")):
        print(f"错误：{yaml_path} 不是YAML文件（需以.yaml或.yml结尾）")
        return None

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            openapi_data = yaml.safe_load(f)

        # 基础校验OpenAPI版本
        if not openapi_data.get('openapi', '').startswith('3.'):
            print(f"警告：当前文件是OpenAPI {openapi_data.get('openapi', '未知')} 版本，建议使用3.x版本")

        # 检查是否包含核心的paths字段（无paths则无API信息）
        if 'paths' not in openapi_data or not openapi_data['paths']:
            print("错误：OpenAPI文件中未找到有效的paths字段（无API端点信息）")
            return None

        print(f"成功解析OpenAPI文件，共包含 {len(openapi_data['paths'])} 个API路径")
        return openapi_data
    except FileNotFoundError:
        print(f"错误：未找到文件 {yaml_path}（请检查路径是否正确）")
        return None
    except yaml.YAMLError as e:
        print(f"错误：YAML格式解析失败 {str(e)}（请检查文件语法）")
        return None
    except Exception as e:
        print(f"错误：解析OpenAPI文件时发生未知错误 {str(e)}")
        return None

def extract_api_info(openapi_data: Dict) -> List[Dict[str, Any]]:
    """提取API信息并生成描述，修复变量引用错误"""
    api_list = []
    paths = openapi_data.get('paths', {})
    components = openapi_data.get('components', {})

    for path, path_details in paths.items():
        # 遍历每个HTTP方法（get/post/put/delete等）
        for method, method_details in path_details.items():
            # 跳过非HTTP方法的字段（如parameters，部分OpenAPI会在path下定义全局参数）
            if method not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                continue

            summary = method_details.get('summary', '无摘要')
            desc = method_details.get('description', '无详细描述')

            try:
                desc_response = description_chain.invoke({
                    "method": method.upper(),
                    "path": path,
                    "summary": summary,
                    "description": desc[:200]  # 截取前200字，避免输入过长
                })
                generated_description = desc_response["text"].strip()
            except Exception:
                print(f"警告：生成 {method.upper()} {path} 的描述失败，使用默认描述")
                generated_description = f"{method.upper()} {path}：{summary[:50]}"  # 默认描述

            # 解析请求体（处理$ref引用）
            request_body = method_details.get('requestBody', {})
            if request_body and "$ref" in request_body:
                ref_key = request_body["$ref"].split('/')[-1]
                request_body = components.get('requestBodies', {}).get(ref_key, request_body)

            api_info = {
                "path": path,
                "method": method.upper(),
                "summary": summary,
                "original_description": desc,
                "generated_description": generated_description,  # 已正确定义的变量
                "tags": method_details.get('tags', []),
                "parameters": method_details.get('parameters', []),
                "request_body": request_body,
                "responses": method_details.get('responses', {}),
                "search_text": f"{generated_description} | 标签：{','.join(method_details.get('tags', []))}"
            }
            print(api_info["search_text"])
            api_list.append(api_info)

    print(f"成功提取 {len(api_list)} 个API端点信息")
    return api_list

def store_apis_in_chromadb(collection: chromadb.Collection, api_list: List[Dict[str, Any]]) -> bool:
    """将API信息存储到ChromaDB，修复删除逻辑"""
    try:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            print(f"成功删除集合中 {len(all_ids)} 条旧数据")

        ids = [str(uuid.uuid4()) for _ in api_list]  # 生成唯一ID
        documents = [api["search_text"] for api in api_list]  # 用于向量搜索的文本，search_text字段是chromaDB默认检索字段
        metadatas = [
            {
                "path": api["path"],
                "method": api["method"],
                "tags": ",".join(api["tags"]),  # 标签转为字符串，方便筛选
                "summary": api["summary"][:100]  # 截取摘要，避免元数据过长
            } for api in api_list
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        global api_id_mapping
        api_id_mapping = {ids[i]: api_list[i] for i in range(len(api_list))}

        print(f"成功将 {len(api_list)} 个API信息存入ChromaDB")
        return True
    except Exception as e:
        print(f"错误：存储API到ChromaDB失败 {str(e)}")
        return False

def search_apis(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """搜索相关API，返回完整信息"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "distances"]  # 包含元数据和相似度距离
        )

        matched_apis = []
        for i in range(len(results["ids"][0])):
            api_id = results["ids"][0][i]
            api_info = api_id_mapping.get(api_id, None)
            if not api_info:
                continue
            api_info["similarity_score"] = round(1 - results["distances"][0][i], 3)  # 转为0-1的相似度
            matched_apis.append(api_info)
        return matched_apis
    except Exception as e:
        print(f"错误：搜索API失败 {str(e)}")
        return []


def format_api_response(apis: List[Dict[str, Any]]) -> str:
    """格式化API响应，使其更易读"""
    if not apis:
        return "未找到与查询相关的API信息，请尝试调整关键词（如API路径、功能描述）"

    response = []
    for idx, api in enumerate(apis, 1):
        if api['similarity_score'] < -10:
            continue
        # 基础信息
        base_info = [
            f"【API {idx}】",
            f"路径：{api['path']}",
            f"方法：{api['method']}",
            f"描述：{api['generated_description']}",
            f"相似度：{api['similarity_score']}",
            f"标签：{','.join(api['tags']) if api['tags'] else '无标签'}",
            f"原始数据：{api}"
        ]

        # 参数信息（只显示前3个，避免过长）
        params = api["parameters"]
        if params:
            base_info.append("\n【参数列表】")
            for p in params:
                param_name = p.get("name", "未知参数")
                param_in = p.get("in", "未知位置")
                param_desc = p.get("description", "无描述")
                base_info.append(f"- {param_in}参数：{param_name} | {param_desc}")

        responses = api["responses"]
        if responses:
            base_info.append("\n【响应示例】")
            common_status = ["200", "201", "400", "401", "404"]
            shown = 0
            for status in common_status:
                if status in responses:
                    resp_desc = responses[status].get("description", "无描述")
                    base_info.append(f"- {status}：{resp_desc}")
                    shown += 1
            other_status = [s for s in responses.keys() if s not in common_status]
            for status in other_status[:2 - shown]:
                resp_desc = responses[status].get("description", "无描述")
                base_info.append(f"- {status}：{resp_desc}")

        response.append("\n".join(base_info) + "\n" + "-" * 50)

    return "\n".join(response)

def initialize_system(yaml_path: str) -> bool:
    print("\n" + "=" * 60)
    print("开始初始化API知识库...")
    # all_ids = collection.get()["ids"]
    # if all_ids:
    #     return True

    openapi_data = load_and_parse_openapi(yaml_path)
    if not openapi_data:
        print("初始化失败：OpenAPI文件解析错误")
        return False

    api_list = extract_api_info(openapi_data)
    if not api_list:
        print("初始化失败：未提取到有效API信息")
        return False

    if not store_apis_in_chromadb(collection, api_list):
        print("初始化失败：API信息存储到ChromaDB错误")
        return False

    print("初始化成功！API知识库已就绪")
    print("=" * 60 + "\n")
    return True


# 全局变量：API ID与完整信息的映射
api_id_mapping = {}


def main():
    print("=" * 70)
    print("        OpenAPI向量知识库查询系统（基于DeepSeek+ChromaDB）        ")
    print("=" * 70)
    print("功能：通过自然语言查询API的路径、方法、参数、响应等信息")
    print("示例：1. 如何获取用户信息？  2. 查找POST请求的API  3. 订单相关接口有哪些？")
    print("退出：输入 'exit' 或 'quit'")
    print("=" * 70)

    # 1. 输入YAML路径并初始化
    while True:
        yaml_path = "api.yaml"
        if yaml_path.lower() in ["exit", "quit"]:
            print("程序退出")
            return
        if initialize_system(yaml_path):
            break
        print("请重新输入正确的YAML文件路径\n")

    # 2. 初始化对话记忆（支持上下文跟进）
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="query",
        return_messages=False,
        human_prefix="用户",
        ai_prefix="助手"
    )

    # 3. 主查询链（优化提示词，让回答更精准）
    main_prompt = PromptTemplate(
        input_variables=["history", "query", "search_results"],
        template="""
        你是API查询助手，需基于以下信息回答用户问题：
        1. 历史对话：{history}（若用户问"它的参数"，需通过历史确定"它"指哪个API）
        2. 搜索结果：{search_results}（这是最准确的API数据，必须基于此回答，不能编造）

        回答规则：
        - 先直接回答用户问题，重点突出API的【路径】和【方法】
        - 若有多个API，按相似度从高到低排序
        - 若搜索结果无相关信息，直接说"未找到相关API"，不要猜测
        - 语言简洁，避免冗余，不添加无关内容

        用户当前问题：{query}
        """
    )

    main_chain = LLMChain(
        llm=llm,
        prompt=main_prompt,
        memory=memory,
        verbose=False
    )

    # 4. 交互循环
    while True:
        user_query = input("\n请输入你的查询（输入exit退出）：").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 程序退出，感谢使用！")
            return
        if not user_query:
            print("⚠️ 请输入有效的查询内容（如'如何创建订单？'）")
            continue

        # 5. 搜索API并生成回答
        print("正在搜索相关API...")
        matched_apis = search_apis(collection, user_query, n_results=10)
        search_results = format_api_response(matched_apis)

        # 6. 调用LLM生成自然语言回答
        try:
            response = main_chain.invoke({
                "query": user_query,
                "search_results": search_results
            })
            print("\n" + "-" * 60)
            print(response["text"].strip())
            print("-" * 60)
        except Exception as e:
            print(f"\n⚠️ 生成回答失败：{str(e)}，直接展示搜索结果：")
            print("-" * 60)
            print(search_results)
            print("-" * 60)


if __name__ == "__main__":
    main()