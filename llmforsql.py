
import sys
import os
import urllib.request  
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
# REMOVED BROKEN IMPORT: from langchain.chains import create_sql_query_chain 
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool # Fixed Capital 'B'
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

LLM_MODEL = "gemma3:1b"
DB_FILE = "Chinook.db"

if not os.path.exists(DB_FILE):
    print("Downloading DB...")
    url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    urllib.request.urlretrieve(url, DB_FILE) # Fixed function call

db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
llm = ChatOllama(model=LLM_MODEL)

# 2. Define Manual SQL Generation (Replaces create_sql_query_chain)
def get_schema(_):
    return db.get_table_info()

sql_prompt = PromptTemplate.from_template(
    """Based on the table schema below, write a SQL query that would answer the user's question.
    Do NOT output any explanation, only the SQL query.
    
    Schema:
    {schema}
    
    Question: {question}
    SQL Query:"""
)

#had some conflicts in packages so using old package
write_query = (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | llm
    | StrOutputParser()
)

def clean_sql(text):
    return text.replace("```sql", "").replace("```", "").strip()

#Execution Tool
execute_query = QuerySQLDataBaseTool(db=db)

#Prompt
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:"""
)

#Pipeline
chain = (
    RunnablePassthrough.assign(query=write_query | clean_sql)
    .assign(result=itemgetter("query") | execute_query)
    | answer_prompt
    | llm
    | StrOutputParser()
)

query = "List 3 songs by AC/DC"
print(f"Query: {query}")

try:
    response = chain.invoke({"question": query})
    print(response)
except Exception as e:
    print(f"Error: {e}")