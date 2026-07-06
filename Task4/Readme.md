# Context-Aware Chatbot Using LangChain / RAG

Build a conversational chatbot that remembers conversation history and retrieves accurate answers from an external knowledge base using Retrieval-Augmented Generation (RAG).

## 🎯 Objective
Plain LLM chatbots can hallucinate and forget earlier context. This project builds a chatbot that (1) retrieves relevant information from a custom document corpus using vector search, and (2) maintains conversational memory, so answers stay grounded and context-aware across turns. The chatbot is deployed as an interactive Streamlit app.

## 🗂️ Dataset / Knowledge Base
Custom corpus — e.g., a set of Wikipedia pages, internal documents, or PDFs relevant to a chosen domain (swap in your own knowledge base as needed).

## 🛠️ Methodology / Approach
1. **Document Ingestion** – Load the custom corpus (text files, PDFs, or scraped Wikipedia pages) and split it into manageable chunks.
2. **Embedding & Vector Store** – Generate embeddings for each chunk (e.g., OpenAI/HuggingFace embeddings) and store them in a vector database (FAISS/Chroma).
3. **Retrieval-Augmented Generation (RAG)** – On each user query, retrieve the most relevant chunks from the vector store and pass them as context to the LLM to generate a grounded answer.
4. **Conversation Memory** – Use LangChain's memory modules (e.g., `ConversationBufferMemory`) to retain prior turns, so follow-up questions are understood in context.
5. **Deployment** – Wrap the chatbot in a Streamlit chat interface for live interaction.

## 📊 Key Results / Observations
- The RAG-based chatbot produced more accurate, source-grounded answers compared to a plain LLM baseline with no retrieval.
- Conversation memory allowed the chatbot to correctly resolve follow-up/pronoun-based questions (e.g., "What about its founder?" after asking about a company).
- Retrieval quality (chunk size, embedding model choice) had a noticeable impact on answer relevance — smaller, well-overlapped chunks performed best for this corpus.

*(Replace with your actual observations/examples after testing your chatbot.)*

## ⚙️ Tools & Technologies
- Python 3
- LangChain
- OpenAI / Anthropic / Hugging Face LLM APIs
- FAISS or ChromaDB (vector store)
- Streamlit (deployment)

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/context-aware-rag-chatbot.git
cd context-aware-rag-chatbot
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Files
| File | Description |
|---|---|
| `Task4_Phase2.ipynb` | Document ingestion, embedding, RAG pipeline development/testing |
| `app.py` | Streamlit chatbot app with conversation memory |
| `data/` | Custom document corpus / knowledge base |
| `README.md` | Project documentation |

## 📌 Skills Demonstrated
- Conversational AI development
- Document embedding and vector search
- Retrieval-Augmented Generation (RAG)
- LLM integration and deployment
