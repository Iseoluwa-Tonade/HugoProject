import streamlit as st
import os
import tempfile

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# --- Cached Function for Processing Documents ---

# Cache the resource so we don't re-process/re-embed on every app rerun for the same files
@st.cache_resource
def get_vector_store(_uploaded_files):
    with st.spinner("Processing documents... this may take a moment."):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = []
            for uploaded_file in _uploaded_files:
                temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Use appropriate loader based on file type
                if temp_filepath.endswith('.pdf'):
                    loader = PyPDFLoader(temp_filepath)
                elif temp_filepath.endswith('.docx'):
                    loader = Docx2txtLoader(temp_filepath)
                elif temp_filepath.endswith('.pptx'):
                    loader = UnstructuredPowerPointLoader(temp_filepath)
                
                docs.extend(loader.load())

        # 1. Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 2. Create vector store from chunks using a local model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        
        return vector_store

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="Query Your Docs", page_icon="📄")
st.title("📄 Query Your Documents")

# Sidebar for API Key and file uploads
with st.sidebar:
    st.header("Configuration")
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("API key loaded from secrets!")
    except:
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, PPTX)",
        type=['pdf', 'docx', 'pptx'],
        accept_multiple_files=True
    )

# Main app logic
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()

# Initialize LangChain Chat Model
try:
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o")
except Exception as e:
    st.error(f"Failed to initialize OpenAI Chat Model: {e}")
    st.stop()

# Initialize session state for vector store
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if uploaded_files:
    if st.button("Process Documents"):
        # Use the cached function to process files
        vector_store = get_vector_store(uploaded_files)
        
        # Create the retrieval chain
        retriever = vector_store.as_retriever()
        prompt_template = ChatPromptTemplate.from_template(
            """Answer the user's question based only on the following context:
            <context>{context}</context>
            Question: {input}"""
        )
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        st.success("Documents processed successfully! You can now ask questions.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.retrieval_chain is None:
        st.warning("Please upload and process documents before asking a question.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.retrieval_chain.invoke({"input": prompt})
                    response_text = response["answer"]
                except Exception as e:
                    response_text = f"An error occurred: {e}"

                st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
