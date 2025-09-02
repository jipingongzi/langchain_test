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

table_info = db.get_table_info()

sql_prompt = PromptTemplate(
    input_variables=["input", "table_info"],
    template="""
        You are a professional SQL query generator, capable of producing correct PostgreSQL queries based on user questions and database table structures.

        ## Database Table Structure:
        {table_info}

        The functions of key tables are as follows:
        consumption_api_desc         ： For API summary or overview  
        all_tickets_info_epmldataai  ： Serves as supplementary table for enriched API information (e.g., business context, logic, request/response examples) in linked Jira ticket 
        api_code_response            ： For code samples or implementation details                                            
        merged_pr_info               ： Contains information about merged pull requests related to a specific API             
        opened_pr_info               ： Stores information about currently opened (unmerged) pull requests for a specific API 

        ## Generation Rules:
        1. Only generate SQL statements that can be executed directly; do not include any explanations, descriptive text, or format markers
        2. Do not include ```sql, ```, or any other markers
        3. Ensure correct table names and column names are used, based on the provided table structure
        4. Only query fields necessary for answering the question
        5. If multi-table query is required, use JOIN correctly for association

        User's Question: {input}

        Generated SQL Statement:
    """
)

answer_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
        You are a data analyst, and you need to answer the user's question based on SQL query results while referring to the historical conversation context.

        ## Historical Conversation (Must be referenced to understand contextual references):
        {history}

        ## Current Query Information (Including user question, executed SQL, query results):
        {input}

        ## Answer Rules:
        1. If the user's question is a follow-up (e.g., "What is its business context?"), you must determine what "it" refers to (the API/content from the previous round) through the historical conversation.
        2. Answer in concise and clear language; do not add extra information.
        3. Provide three possible follow-up questions to help the user gain a deeper understanding of relevant information.
    """
)

llm = ChatOpenAI(
    api_key="sk-9b5776bd68e045f7ae2171077134b2a4",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1
)

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=False,
    ai_prefix="Assistant",
    human_prefix="User"
)

sql_chain = LLMChain(llm=llm, prompt=sql_prompt)
answer_chain = LLMChain(
    llm=llm,
    prompt=answer_prompt,
    memory=memory,
    verbose=False
)


def clean_sql_query(sql):
    if sql.startswith('```sql'):
        sql = sql[len('```sql'):]
    if sql.startswith('```'):
        sql = sql[len('```'):]
    if sql.endswith('```'):
        sql = sql[:-len('```')]
    return sql.strip()


def ask_database(question: str):
    try:
        sql_response = sql_chain.invoke(
            input={
                "input": question,
                "table_info": table_info
            }
        )
        raw_sql = sql_response["text"].strip()
        sql_query = clean_sql_query(raw_sql)

        query_result = db.run(sql_query)

        combined_input = f"""
            User's Question: {question}
            Executed SQL Query: {sql_query}
            Query Result: {query_result}
        """

        answer_response = answer_chain.invoke(
            input={"input": combined_input}
        )
        answer = answer_response["text"]

        return answer

    except Exception as e:
        return f"An error occurred while executing the query: {str(e)}\nPlease check the question description or database connection."


if __name__ == "__main__":
    print("Welcome to the API Knowledge Base Query Tool (enter 'exit' to quit)!")
    while True:
        user_question = input("\nPlease enter your question:")
        if user_question.lower() == "exit":
            print("Program exited.")
            break
        response = ask_database(user_question)
        print("\n" + "-" * 50)
        print(response)
        print("-" * 50)