# AskMate – Ask Anything from Your Documents

AskMate is a smart Q&A web app built with **Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Google Gemini AI**.  
It allows you to upload one or more PDF files, ask questions, and get answers either from the document(s) or from general knowledge if the document does not contain the answer.

---

## Features

- Upload single or multiple PDF files.
- Ask any question related to the documents or general topics.
- Answers come from the PDFs first; fallback to Gemini AI general knowledge if not found.
- Source tracking with file name and page numbers.
- Chat history maintained during the session.
- Option to download chat history as a text file.
- Clear chat history with a single click.
- Debug mode to view retrieved PDF chunks.
- Error handling with retry logic for Gemini API rate limits.

---

## Tech Stack

- **Frontend/Framework**: Streamlit  
- **Document Processing**: LangChain (PyPDFLoader, CharacterTextSplitter)  
- **Vector Store**: ChromaDB  
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)  
- **LLM**: Google Gemini 1.5 Flash  
- **Environment Management**: Python-dotenv  
