import traceback
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from zalando_apidoc_demo import search_apis, format_api_response, initialize_system as init_api_docs
# 导入你的功能模块
from zalando_pg_demo import ask_database


# --------------------------
# 关键：启动时执行API文档初始化
# --------------------------
def initialize_api_collection():
    try:
        api_collection = init_api_docs("api.yaml", )

        if api_collection is not None:
            print("✅ API文档初始化成功，集合对象已获取")
            return api_collection
        else:
            print("❌ API文档初始化完成，但未返回有效的集合对象")
            return None

    except ImportError:
        error_msg = "未找到zalando_apidoc_demo中的main函数，请检查函数名称是否正确"
        print(f"❌ {error_msg}")
        return None
    except Exception as e:
        error_msg = f"API文档初始化失败：{str(e)}\n详细错误：{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return None


# 在程序启动时执行初始化并获取集合
initialize_api_collection()


# 定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]
    database_result: Optional[str] = None
    api_result: Optional[str] = None
    question: str
    next_step: Optional[str] = None


# 判断结果是否有效
def is_valid_result(result: Optional[str]) -> bool:
    if result is None or result.strip() == "":
        return False
    no_result_phrases = ["无结果", "未找到", "不存在", "没有", "null", "none", "错误"]
    return not any(phrase in result.lower() for phrase in no_result_phrases)


# 1. 查询数据库节点
def query_database(state: State) -> State:
    question = state["question"]
    print(f"🔍 正在数据库中查询: {question}")

    try:
        db_result = ask_database(question)
        if not db_result or db_result.strip() == "":
            db_result = "数据库中未找到相关数据"
    except Exception as e:
        db_result = f"数据库查询失败：{str(e)}"

    return {
        **state,
        "database_result": db_result,
        "api_result": None
    }


# 2. 决策节点
def decide_next_step(state: State) -> State:
    if is_valid_result(state["database_result"]):
        return {**state, "next_step": "generate_response"}
    else:
        return {**state, "next_step": "query_api"}


# 3. 查询API节点
def query_api(state: State) -> State:
    question = state["question"]
    print(f"🔍 数据库未找到结果，正在API文档中查询: {question}")

    try:
        # 调用API搜索
        api_matches = search_apis(query=question, n_results=5)
        if not api_matches or len(api_matches) == 0:
            api_result = f"API文档中未找到与「{question}」相关的信息"
        else:
            api_result = format_api_response(api_matches)

        return {**state, "api_result": api_result}

    except Exception as e:
        error_detail = traceback.format_exc().split("\n")[-5:]
        error_msg = f"API搜索失败：{str(e)}\n错误详情：{''.join(error_detail)}"
        print(f"❌ {error_msg}")
        return {**state, "api_result": error_msg}


# 4. 生成最终回答节点
def generate_response(state: State) -> State:
    question = state["question"]
    db_result = state.get("database_result", "未查询数据库")
    api_result = state.get("api_result", "未查询API文档")

    print("-----------------------------")
    print(db_result)
    print(api_result)
    print("-----------------------------")

    if is_valid_result(db_result):
        response = f"✅ 关于「{question}」的查询结果（来自数据库）：\n\n{db_result}"
    elif is_valid_result(api_result):
        response = f"✅ 关于「{question}」的查询结果（来自API文档）：\n\n{api_result}"
    else:
        response = f"❌ 未找到关于「{question}」的有效信息：\n- 数据库：{db_result.strip()}\n- API文档：{api_result.strip()}"

    return {**state, "messages": state["messages"] + [AIMessage(content=response)]}


# 构建LangGraph流程
builder = StateGraph(State)
builder.add_node("query_database", query_database)
builder.add_node("decide_next_step", decide_next_step)
builder.add_node("query_api", query_api)
builder.add_node("generate_response", generate_response)

builder.set_entry_point("query_database")
builder.add_edge("query_database", "decide_next_step")
builder.add_conditional_edges(
    "decide_next_step",
    lambda state: state["next_step"],
    {"generate_response": "generate_response", "query_api": "query_api"}
)
builder.add_edge("query_api", "generate_response")
builder.add_edge("generate_response", END)

# 编译图
app = builder.compile()


# 对外调用函数
def run_priority_agent(question: str) -> str:
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "next_step": None,
        "database_result": None,
        "api_result": None
    }
    result = app.invoke(initial_state)
    return result["messages"][-1].content


# 测试代码
if __name__ == "__main__":
    # 测试问题
    test_questions = [
        "查询订单相关的API",
        "剪辑相关API？",
        "查询火箭发射的时间"
    ]

    for q in test_questions:
        print(f"\n{'=' * 50}")
        print(f"用户问题: {q}")
        print(f"回答: {run_priority_agent(q)}")
        print(f"{'=' * 50}\n")
