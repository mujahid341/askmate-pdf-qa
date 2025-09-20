
# import os
# import time
# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.api_core.exceptions

# # Load environment variables and configure API key
# load_dotenv()
# try:
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# except Exception:
#     st.error("Failed to configure API. Please check your key in .env file.")
#     st.stop()

# # Streamlit page setup
# st.set_page_config(
#     page_title="AskMate – Ask Anything from Your Document",
#     layout="centered",
# )

# st.markdown(
#     """
#     <h2 style="text-align:center; color:#2C3E50; margin-bottom:0;">
#         AskMate – Ask Anything from Your Document
#     </h2>
#     <p style="text-align:center; color:gray; margin-top:0;">
#         Upload a PDF and ask questions. Answers will come from the document first, or general knowledge if not found.
#     </p>
#     """,
#     unsafe_allow_html=True,
# )

# # Retry logic for API calls (429 errors)
# def safe_invoke(func, *args, **kwargs):
#     for attempt in range(5):
#         try:
#             return func(*args, **kwargs)
#         except google.api_core.exceptions.ResourceExhausted:
#             wait = 2 ** attempt
#             st.warning(f"Rate limit hit. Retrying in {wait} seconds...")
#             time.sleep(wait)
#     st.error("Could not get answer after retries.")
#     return None

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "pdf_ready" not in st.session_state:
#     st.session_state.pdf_ready = False

# # File uploader
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# if uploaded_file:
#     try:
#         pdf_path = "temp.pdf"
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.success(f"File uploaded successfully: {uploaded_file.name}")
#         st.session_state.pdf_ready = True
#     except Exception as e:
#         st.error(f"Error saving PDF: {e}")
#         st.stop()

# if not st.session_state.pdf_ready:
#     st.warning("Please upload a PDF file before asking a question.")

# if st.session_state.pdf_ready:
#     # Load and split PDF
#     try:
#         with st.spinner("Reading and preparing your document..."):
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#             chunks = splitter.split_documents(documents)
#     except Exception as e:
#         st.error(f"Error processing PDF: {e}")
#         st.stop()

#     # Create local vector index
#     try:
#         with st.spinner("Indexing content..."):
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",  # stronger embeddings
#                 model_kwargs={"device": "cpu"}
#             )
#             vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
#     except Exception as e:
#         st.error(f"Error creating search index: {e}")
#         st.stop()

#     # Load model
#     try:
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
#         retriever = vectorstore.as_retriever()

#         # Question input and button
#         query = st.text_input("Ask a question from the document or general topic:")
#         if st.button("Get Answer"):
#             if query.strip():
#                 with st.spinner("Generating answer..."):
#                     try:
#                         #  Try retrieving from PDF
#                         retrieved_docs = retriever.get_relevant_documents(query)

#                         answer = ""
#                         source_info = ""

#                         if retrieved_docs:  # Found some chunks
#                             context = "\n\n".join([doc.page_content for doc in retrieved_docs])
#                             prompt = (
#                                 "Use the following PDF context to answer if possible. "
#                                 "If the context is irrelevant, just reply with 'NO_CONTEXT'.\n\n"
#                                 f"{context}\n\nQuestion: {query}"
#                             )
#                             response = safe_invoke(llm.invoke, prompt)
#                             answer = response.content.strip() if response else ""

#                             # Detect irrelevant context  fallback
#                             if (
#                                 not answer
#                                 or "NO_CONTEXT" in answer
#                                 or answer.lower().startswith("this question cannot be answered")
#                             ):
#                                 answer = ""  # trigger fallback

#                         # If no good answer fallback to general knowledge
#                         if not answer:
#                             response = safe_invoke(llm.invoke, query)
#                             answer = response.content if response else "Could not get a valid answer."
#                             source_info = "General Knowledge"
#                         else:
#                             pages = sorted(set([doc.metadata.get("page", "Unknown") for doc in retrieved_docs]))
#                             source_info = f"PDF (pages {', '.join(map(str, pages))})"

#                             # Debug: Show retrieved chunks
#                             with st.expander("Retrieved PDF Chunks (debug)"):
#                                 for i, doc in enumerate(retrieved_docs, 1):
#                                     st.write(f"Chunk {i} (Page {doc.metadata.get('page', '?')}):")
#                                     st.write(doc.page_content[:500] + "...")

#                         # Save chat history
#                         st.session_state.chat_history.append(
#                             {"question": query, "answer": answer, "source": source_info}
#                         )
#                         st.success("Answer generated successfully.")

#                     except Exception as e:
#                         st.error(f"Error generating answer: {e}")
#             else:
#                 st.warning("Please enter a question before clicking the button.")

#         # Display chat history
#         if st.session_state.chat_history:
#             st.subheader("Chat History")
#             for i, chat in enumerate(st.session_state.chat_history, 1):
#                 st.markdown(
#                     f"""
#                     <div style="padding:12px; margin-bottom:10px; background-color:#F9F9F9; border-radius:8px; border:1px solid #DDD;">
#                         <b>Q{i}:</b> {chat['question']}  
#                         <br><br>
#                         <b>A{i}:</b> {chat['answer']}  
#                         <br><br>
#                         <span style="color:#1A73E8; font-weight:bold;">Source: {chat['source']}</span>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#             # Download chat history
#             history_text = "\n\n".join(
#                 [
#                     f"Q{i}: {chat['question']}\nA{i}: {chat['answer']}\nSource: {chat['source']}"
#                     for i, chat in enumerate(st.session_state.chat_history, 1)
#                 ]
#             )
#             st.download_button(
#                 label="Download Chat History",
#                 data=history_text,
#                 file_name="AskMate_Chat_History.txt",
#                 mime="text/plain",
#             )

#     except Exception as e:
#         st.error(f"Failed to initialize app components: {e}")



import os
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.api_core.exceptions

# Load environment variables and configure API key
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    st.error("Failed to configure API. Please check your key in .env file.")
    st.stop()

# Streamlit page setup
st.set_page_config(
    page_title="AskMate – Ask Anything from Your Documents",
    layout="centered",
)

st.markdown(
    """
    <h2 style="text-align:center; color:#2C3E50; margin-bottom:0;">
        AskMate – Ask Anything from Your Documents
    </h2>
    <p style="text-align:center; color:gray; margin-top:0;">
        Upload one or more PDFs and ask questions. Answers will come from the documents first, or general knowledge if not found.
    </p>
    """,
    unsafe_allow_html=True,
)

# Retry logic for API calls (429 errors)
def safe_invoke(func, *args, **kwargs):
    for attempt in range(5):
        try:
            return func(*args, **kwargs)
        except google.api_core.exceptions.ResourceExhausted:
            wait = 2 ** attempt
            st.warning(f"Rate limit hit. Retrying in {wait} seconds...")
            time.sleep(wait)
    st.error("Could not get answer after retries.")
    return None

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# File uploader - multiple PDFs supported
uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    try:
        documents = []
        for uploaded_file in uploaded_files:
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File uploaded successfully: {uploaded_file.name}")

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Add filename to metadata for source tracking
            for d in docs:
                d.metadata["filename"] = uploaded_file.name

            documents.extend(docs)

        st.session_state.pdf_ready = True

    except Exception as e:
        st.error(f"Error saving/processing PDFs: {e}")
        st.stop()

if not st.session_state.pdf_ready:
    st.warning("Please upload at least one PDF file before asking a question.")

if st.session_state.pdf_ready:
    # Split all documents together
    try:
        with st.spinner("Reading and preparing your documents..."):
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error splitting documents: {e}")
        st.stop()

    # Create one combined index for all PDFs
    try:
        with st.spinner("Indexing all content..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating search index: {e}")
        st.stop()

    # Load model
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        retriever = vectorstore.as_retriever()

        # Question input and button
        query = st.text_input("Ask a question from the documents or general topic:")
        col1, col2 = st.columns([1, 1])
        ask_button = col1.button("Get Answer")
        clear_button = col2.button("Clear History")

        if clear_button:
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

        if ask_button:
            if query.strip():
                with st.spinner("Generating answer..."):
                    try:
                        retrieved_docs = retriever.get_relevant_documents(query)
                        answer = ""
                        source_info = ""

                        if retrieved_docs:
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            prompt = (
                                "Use the following PDF context to answer if possible. "
                                "If the context is irrelevant, just reply with 'NO_CONTEXT'.\n\n"
                                f"{context}\n\nQuestion: {query}"
                            )
                            response = safe_invoke(llm.invoke, prompt)
                            answer = response.content.strip() if response else ""

                            if (
                                not answer
                                or "NO_CONTEXT" in answer
                                or answer.lower().startswith("this question cannot be answered")
                            ):
                                answer = ""

                        if not answer:
                            response = safe_invoke(llm.invoke, query)
                            answer = response.content if response else "Could not get a valid answer."
                            source_info = "General Knowledge"
                        else:
                            pages = sorted(set([doc.metadata.get("page", "?") for doc in retrieved_docs]))
                            files = sorted(set([doc.metadata.get("filename", "Unknown") for doc in retrieved_docs]))
                            source_info = f"PDF ({', '.join(files)} | pages {', '.join(map(str, pages))})"

                            with st.expander("Retrieved PDF Chunks (debug)"):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"Chunk {i} (File {doc.metadata.get('filename', '?')} | Page {doc.metadata.get('page', '?')}):")
                                    st.write(doc.page_content[:500] + "...")

                        st.session_state.chat_history.append(
                            {"question": query, "answer": answer, "source": source_info}
                        )
                        st.success("Answer generated successfully.")

                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.warning("Please enter a question before clicking the button.")

        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(
                    f"""
                    <div style="padding:12px; margin-bottom:10px; background-color:#F9F9F9; border-radius:8px; border:1px solid #DDD;">
                        <b>Q{i}:</b> {chat['question']}  
                        <br><br>
                        <b>A{i}:</b> {chat['answer']}  
                        <br><br>
                        <span style="color:#1A73E8; font-weight:bold;">Source: {chat['source']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            history_text = "\n\n".join(
                [
                    f"Q{i}: {chat['question']}\nA{i}: {chat['answer']}\nSource: {chat['source']}"
                    for i, chat in enumerate(st.session_state.chat_history, 1)
                ]
            )
            st.download_button(
                label="Download Chat History",
                data=history_text,
                file_name="AskMate_Chat_History.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Failed to initialize app components: {e}")
