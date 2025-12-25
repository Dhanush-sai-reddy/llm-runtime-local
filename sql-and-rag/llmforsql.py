import os
import urllib.request
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

LLM_MODEL="gemma3:1b"
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_FILE=os.path.join(BASE_DIR, "company.db")

if not os.path.exists(DB_FILE):
    url="https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    urllib.request.urlretrieve(url, DB_FILE)

db=SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
llm=ChatOllama(model=LLM_MODEL)

def get_schema(_):
    return db.get_table_info()

def clean_sql(text):
    return text.replace("```sql", "").replace("```", "").strip()

sql_prompt=PromptTemplate.from_template(
    """Based on schema, write SQL query. No explanation.
    Schema: {schema}
    Question: {question}
    SQL Query:"""
)

write_query=(
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | llm
    | StrOutputParser()
)

execute_query=QuerySQLDataBaseTool(db=db)

answer_prompt=PromptTemplate.from_template(
    """Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:"""
)

chain=(
    RunnablePassthrough.assign(query=write_query | clean_sql)
    .assign(result=itemgetter("query") | execute_query)
    | answer_prompt
    | llm
    | StrOutputParser()
)

def ask_sql(question):
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        return f"Error: {e}"

if __name__=="__main__":
    print(ask_sql("List 3 songs by AC/DC"))