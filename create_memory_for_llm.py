from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#step-1 load raw pdf(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',         #for all pdfs available
                           loader_cls=PyPDFLoader)
    documents=loader.load()  #returns the pages of the pdf
    return documents
documentsp=load_pdf_files(data=DATA_PATH)
#print("length of the PDF is:",len(documentsp))

#step-2 create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,  #words per chunk
                                                 chunk_overlap=50) #buffer context infact!!
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documentsp)
#print("length of txt  chunks:",len(text_chunks))

#step-3 Vector Embeddings

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return embedding_model

embedding_model=get_embedding_model()

#step-4 store embd in FAISS

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
print("check")