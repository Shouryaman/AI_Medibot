🩺 MediBot: AI Medical Assistant with RAG using Gemini, FAISS & Streamlit

A Retrieval-Augmented Generation (RAG) Medical Bot using Gemini models for generation, FAISS for retrieval, and Streamlit for a clean UI. It provides grounded, accurate medical insights from your knowledge base while maintaining a friendly conversational flow.


---

🚀 Features

✅ Gemini + FAISS RAG pipeline
✅ Accurate medical Q&A with context grounding
✅ Upload your medical PDFs to build memory
✅ Fast semantic search with Hugging Face embeddings + FAISS
✅ Streamlit chat UI for easy interaction
✅ Modular, extensible code for your medical projects


---

🛠 Tech Stack

Gemini LLM (Google Generative AI)

FAISS vector store

Hugging Face Embeddings

Streamlit

Python 3.11+



---

🗂 Project Structure

📦 medi-rag-bot
 ┣ 📂 vectorstore/
 ┣ 📂 data/
 ┣ 📜 create_memory_for_llm.py   # For building vector memory
 ┣ 📜 medibot.py                 # Streamlit chat UI
 ┣ 📜 requirements.txt
 ┗ 📜 README.md


---

⚡ Quickstart

1️⃣ Clone the repository:

git clone https://github.com/yourusername/medi-rag-bot.git
cd medi-rag-bot

2️⃣ Install dependencies:

pip install -r requirements.txt

3️⃣ Set environment variables:

GOOGLE_API_KEY for Gemini access

(Optional) Use a .env file for local management


4️⃣ Create vector memory:

Place your medical PDFs into data/ and run:

python create_memory_for_llm.py

This will generate your FAISS vector database under vectorstore/.

5️⃣ Run MediBot:

streamlit run medibot.py

Open http://localhost:8501 in your browser to start chatting with MediBot.


---

🧩 How It Works

1. User inputs a medical question.


2. Embeddings are generated and matched in FAISS for relevant context.


3. Top matching chunks are sent to Gemini as context for grounded generation.


4. Gemini generates precise, context-rich medical responses.


5. Response is displayed in Streamlit chat UI.




---

📈 Roadmap

✅ Multi-document ingestion support (PDF, TXT)

✅ RAG optimization for faster response

[ ] Add references/citations in responses

[ ] Authentication layer for privacy

[ ] Deployment on Streamlit Community Cloud or Hugging Face Spaces



---

🤝 Contributing

Open issues or pull requests to improve MediBot, add features, or refine the UI.


---

⚠ Disclaimer

MediBot is intended for educational and informational purposes only and does not replace professional medical advice. Always consult a qualified healthcare provider for serious medical conditions.
