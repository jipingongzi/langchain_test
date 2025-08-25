from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

db = SQLDatabase.from_uri(
    "postgresql+psycopg2://postgres:123456@51.21.108.101:54322/xuanfeng",
    include_tables=["consumption_api_desc", "all_tickets_info_epmldataai",
                    "api_code_response", "merged_pr_info", "opened_pr_info"]
)

# 获取数据库表结构信息
table_info = db.get_table_info()

# 创建生成SQL的Prompt - 修复变量引用错误
sql_prompt = PromptTemplate(
    input_variables=["input", "table_info"],  # 明确声明两个输入变量
    template="""
        你是一个专业的SQL查询生成器，能根据用户问题和数据库表结构生成正确的PostgreSQL查询语句。
        
        ## 数据库表结构:
        {table_info}
        
        主要表的作用如下：
        consumption_api_desc         ： For API summary or overview  
        all_tickets_info_epmldataai  ： Serves as supplementary table for enriched API information (e.g., business context, logic, request/response examples) in linked Jira ticket 
        api_code_response            ： For code samples or implementation details                                            
        merged_pr_info               ： Contains information about merged pull requests related to a specific API             
        opened_pr_info               ： Stores information about currently opened (unmerged) pull requests for a specific API 
        
        ## 生成规则:
        1. 仅生成可直接执行的SQL语句，不要有任何解释、说明文字或格式标记
        2. 不要包含```sql、```或任何其他标记
        3. 确保使用正确的表名和列名，基于提供的表结构
        4. 只查询回答问题所必需的字段
        5. 如果需要多表查询，请正确使用JOIN关联
        
        用户的问题: {input}
        
        生成的SQL语句:
    """
)

# 创建解释查询结果的Prompt
answer_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
        你是一个数据分析师，需要结合历史对话上下文，根据SQL查询结果回答用户的问题。
        
        ## 历史对话（必须参考，理解上下文指代）:
        {history}
        
        ## 当前查询信息（包含用户问题、执行的SQL、查询结果）:
        {input}
        
        ## 回答规则:
        1. 如果用户的问题是追问（如“它的业务上下文是什么”），必须通过历史对话确定“它”指的是上一轮的API/内容。
        2. 用简洁明了的语言回答，不要添加额外信息。
        3. 提供三个可能的后续问题，帮助用户进一步了解相关信息。
    """
)

# 初始化LLM
llm = ChatOpenAI(
    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1
)

# 创建带记忆的链
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",  # 匹配answer_prompt的input变量
    return_messages=False,
    ai_prefix="助手",
    human_prefix="用户"
)

# 创建链
sql_chain = LLMChain(llm=llm, prompt=sql_prompt)
answer_chain = LLMChain(
    llm=llm,
    prompt=answer_prompt,
    memory=memory,
    verbose=False
)

def clean_sql_query(sql):
    """清理SQL查询中的Markdown格式标记"""
    if sql.startswith('```sql'):
        sql = sql[len('```sql'):]
    if sql.startswith('```'):
        sql = sql[len('```'):]
    if sql.endswith('```'):
        sql = sql[:-len('```')]
    return sql.strip()


def ask_database(question: str):
    try:
        # 生成SQL查询 - 正确传递两个输入变量
        sql_response = sql_chain.invoke(
            input={
                "input": question,  # 用户问题
                "table_info": table_info  # 表结构信息
            }
        )
        raw_sql = sql_response["text"].strip()
        sql_query = clean_sql_query(raw_sql)
        print(f"生成的SQL查询: {sql_query}")

        # 执行SQL查询
        query_result = db.run(sql_query)
        print(f"查询结果: {query_result}")

        # 拼接信息为单输入变量
        combined_input = f"""
            用户的问题: {question}
            执行的SQL查询: {sql_query}
            查询结果: {query_result}
        """

        # 生成自然语言回答
        answer_response = answer_chain.invoke(
            input={"input": combined_input}
        )
        answer = answer_response["text"]

        return answer

    except Exception as e:
        return f"执行查询时发生错误: {str(e)}\n请检查问题描述或数据库连接。"


if __name__ == "__main__":
    print("欢迎使用API知识库查询工具（输入 'exit' 退出）！")
    print("示例流程：\n1. 先问'API_X是什么？'\n2. 再追问'它的业务上下文是什么'")
    while True:
        user_question = input("\n请输入你的问题：")
        if user_question.lower() == "exit":
            print("已退出程序。")
            break
        response = ask_database(user_question)
        print("\n" + "-" * 50)
        print(response)
        print("-" * 50)
