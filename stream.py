import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv  # Import dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models
from langchain.schema import HumanMessage     # Import HumanMessage
# Load environment variables from .env file
# load_dotenv()
# # Set OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Use Streamlit secrets in production
# Use Streamlit secrets in production; fallback to .env for local development
if "general" in st.secrets and "OPENAI_API_KEY" in st.secrets["general"]:
    OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]
st.set_page_config(layout="wide")
# Initialize session state for retriever
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.title("🔎 Web Content Retriever with FAISS")

# Input field for URL
url = st.text_input("Enter the webpage URL:", "")

# Function to create retriever
def create_retriever(url):
    try:
        st.write("📂 Loading webpage content...") 
        loader = WebBaseLoader(url)
        docs = loader.load()

        if not docs:
            st.error("⚠️ No content found on this webpage!")
            return None

        # Splitting text
        st.write("✂️ Splitting text into chunks...") 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        if not documents:
            st.error("⚠️ Failed to split text into chunks.")
            return None

        # Creating FAISS vector store
        st.write("🧠 Creating FAISS vector store...")
        vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

        return vectordb.as_retriever()
    
    except Exception as e:
        st.error(f"❌ Error processing URL: {e}")
        return None


if st.button("Process URL"):
    if url:
        with st.spinner("⏳ Processing webpage..."):
            retriever = create_retriever(url)
        
        if retriever:
            st.session_state.retriever = retriever  # Store retriever in session state
            st.success("✅ Webpage processed successfully!")
        else:
            st.error("⚠️ Failed to process webpage. Please check the URL and try again.")

# Input field for search query
query = st.text_input("Enter your search query:", "")

# Search button
if st.button("Search"):
    if st.session_state.retriever is None:
        st.warning("⚠️ Please process a webpage first!")
    elif not query:
        st.warning("⚠️ Please enter a search query!")
    else:
        with st.spinner("🔍 Searching..."):
            results = st.session_state.retriever.get_relevant_documents(query, k=5)  # Retrieve top 5 chunks

        if results:
            context = "\n\n".join([doc.page_content for doc in results])  
            
  
            print("\n\n📜 Retrieved Chunks (Terminal Output):\n")
            for i, doc in enumerate(results):
                print(f"🔹 Chunk {i+1}:\n{doc.page_content}\n{'-'*50}")

            # Call LLM with the retrieved context using ChatOpenAI
            with st.spinner("🤖 Generating response..."):
                llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
                prompt = f"""
            You are a URL Info AI Assistant. The following context has been extracted from a webpage. Based on this context, please answer the query below as accurately as possible. 
            If the answer is not fully covered by the context, provide your best response while clearly noting that the answer is based on limited webpage information.
                
            Query: {query}

            Context:
            {context}

            Please provide a clear, concise, and helpful answer.
            """
                response = llm([HumanMessage(content=prompt)])

            
            st.subheader("🤖 LLM Response:")
            st.write(response.content)
        else:
            st.warning("❌ No relevant results found.")
