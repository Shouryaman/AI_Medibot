import os
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------- ENV ----------------------
load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your environment.")

# ---------------------- LLM Loader ----------------------

def load_llm_gemini():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    return llm

# ---------------------- Prompt Template ----------------------

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know. Do not make up an answer.
Do not provide anything outside the given context.

Context:
{context}

Question:
{question}

Start your answer directly, no small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# ---------------------- Load FAISS Vector Store ----------------------

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ---------------------- Build QA Chain ----------------------

llm = load_llm_gemini()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

# ---------------------- User Query Execution ----------------------

user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:")
for idx, doc in enumerate(response["source_documents"]):
    print(f"\nDocument {idx + 1}:\n{doc.page_content}\n")