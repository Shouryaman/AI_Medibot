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

# ----------------- FAISS Loader -----------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ----------------- Prompt Setup -----------------
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# ----------------- Gemini Loader -----------------
def load_llm_gemini():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    return llm

# ----------------- Main App -----------------
def main():
    st.title("🩺 Ask MediBot!")

    llm = load_llm_gemini()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    CUSTOM_PROMPT_TEMPLATE = """
    Use ONLY the pieces of information provided in the context below to answer the user's question.
    If you do not know the answer based on the context, respond with "I do not know."
    Do not provide any information outside the provided context.

    Context:
    {context}

    Question:
    {question}

    Start your answer directly.
    """
    prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt_template}
    )

    # ----------------- Chat UI -----------------
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.spinner("MediBot is thinking..."):
            try:
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_docs = response.get("source_documents", [])

                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

                if source_docs:
                    with st.expander("📂 Show Sources Used in Answer"):
                        for idx, doc in enumerate(source_docs):
                            snippet = doc.page_content[:500].strip().replace('\n', ' ')
                            st.markdown(f"[{idx + 1}] Source Snippet:** {snippet}...")
                            if doc.metadata:
                                st.markdown(f"*Metadata:* {doc.metadata}")
                            st.markdown("---")
                else:
                    st.info("No sources were retrieved for this answer.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
