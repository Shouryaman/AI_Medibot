import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm_gemini():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    return llm

def main():
    st.title("Ask Medibot!")

    # Load Gemini LLM and FAISS vectorstore
    llm = load_llm_gemini()
    db = get_vectorstore()
    retriever = db.as_retriever()

    # Custom prompt
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
    prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    # Create RetrievalQA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.spinner("Thinking..."):
            response = qa_chain.invoke({'query': prompt})['result']

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()