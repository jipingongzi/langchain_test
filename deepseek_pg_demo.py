import os
from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 连接 PostgreSQL 数据库
db = SQLDatabase.from_uri(
    "postgresql+psycopg2://postgres:123456@51.21.108.101:54322/xuanfeng",
    include_tables=["consumption_api_desc", "all_tickets_info_epmldataai",
                    "api_code_response", "merged_pr_info", "opened_pr_info"]
)

# 获取数据库表结构信息
table_info = db.get_table_info()

# 创建生成SQL的Prompt - 增强提示以避免格式标记
sql_prompt = PromptTemplate(
    input_variables=["question", "table_info"],
    template="""
你是一个专业的SQL查询生成器，能根据用户问题和数据库表结构生成正确的PostgreSQL查询语句。

## 数据库表结构:
{table_info}

## 生成规则:
1. 仅生成可直接执行的SQL语句，不要有任何解释、说明文字或格式标记
2. 不要包含```sql、```或任何其他标记
3. 确保使用正确的表名和列名，基于提供的表结构
4. 只查询回答问题所必需的字段
5. 如果需要多表查询，请正确使用JOIN关联

用户的问题: {question}

生成的SQL语句:
"""
)

# 创建解释查询结果的Prompt
answer_prompt = PromptTemplate(
    input_variables=["question", "sql_query", "query_result"],
    template="""
你是一个数据分析师，需要根据SQL查询结果回答用户的问题。

用户的问题: {question}
执行的SQL查询: {sql_query}
查询结果: {query_result}

请用简洁明了的语言回答用户的问题，不要添加额外信息。
然后提供三个可能的后续问题，帮助用户进一步了解相关信息。
"""
)

# 初始化LLM
llm = ChatOpenAI(
    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1
)

# 创建链
sql_chain = LLMChain(llm=llm, prompt=sql_prompt)
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)


def clean_sql_query(sql):
    """清理SQL查询中的Markdown格式标记"""
    # 移除开头的```sql标记
    if sql.startswith('```sql'):
        sql = sql[len('```sql'):]
    # 移除开头的```标记
    if sql.startswith('```'):
        sql = sql[len('```'):]
    # 移除结尾的```标记
    if sql.endswith('```'):
        sql = sql[:-len('```')]
    # 去除前后空格
    return sql.strip()


def ask_database(question: str):
    try:
        # 生成SQL查询
        raw_sql = sql_chain.run(question=question, table_info=table_info).strip()
        # 清理SQL查询，移除Markdown格式
        sql_query = clean_sql_query(raw_sql)
        print(f"生成的SQL查询: {sql_query}")

        # 执行SQL查询
        query_result = db.run(sql_query)
        print(f"查询结果: {query_result}")

        # 生成自然语言回答
        answer = answer_chain.run(
            question=question,
            sql_query=sql_query,
            query_result=query_result
        )

        return answer

    except Exception as e:
        return f"执行查询时发生错误: {str(e)}\n请检查问题描述或数据库连接。"


# 交互部分
if __name__ == "__main__":
    print("欢迎使用API知识库查询工具（输入 'exit' 退出）！")
    print("示例问题：'API_X是什么？' 或 '查询API_Y的业务上下文'")
    while True:
        user_question = input("\n请输入你的问题：")
        if user_question.lower() == "exit":
            print("已退出程序。")
            break
        response = ask_database(user_question)
        print("\n" + "-" * 50)
        print(response)
        print("-" * 50)
