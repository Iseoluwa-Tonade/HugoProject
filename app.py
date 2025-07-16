import streamlit as st
import os
import tempfile

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain.memory import ConversationBufferMemory                 # <-- NEW: Import memory
from langchain.chains import create_history_aware_retriever         # <-- NEW: Import history aware retriever
from langchain_core.messages import HumanMessage, AIMessage         # <-- NEW: Import message types

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="Conversational Document AI", page_icon="🧠")
st.title("🧠 Conversational Document AI")

# --- Configuration ---
with st.sidebar:
    st.header("Configuration")
    try:
        # Load secrets for deployment
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
        st.success("API keys loaded from secrets!")
    except:
        # Fallback for local testing
        OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
        PINECONE_API_KEY = st.text_input("Enter your Pinecone API Key:", type="password")
        PINECONE_ENVIRONMENT = st.text_input("Enter your Pinecone Environment:")

    PINECONE_INDEX_NAME = st.text_input("Enter your Pinecone Index Name:", value="rag-documents")
    
    uploaded_files = st.file_uploader(
        "Upload new documents to add to the memory",
        type=['pdf', 'docx', 'pptx'],
        accept_multiple_files=True
    )

# --- Main App Logic ---
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    st.warning("Please provide all required configurations in the sidebar.")
    st.stop()

# Initialize models and embeddings
try:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    retriever = vector_store.as_retriever()
except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    st.stop()

# --- Logic to Add New Documents to Pinecone ---
if uploaded_files:
    if st.button("Add Documents to Memory"):
        with st.spinner("Processing documents and adding to permanent memory..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                docs = []
                for uploaded_file in uploaded_files:
                    temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_filepath, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    if temp_filepath.endswith('.pdf'):
                        loader = PyPDFLoader(temp_filepath)
                    elif temp_filepath.endswith('.docx'):
                        loader = Docx2txtLoader(temp_filepath)
                    elif temp_filepath.endswith('.pptx'):
                        loader = UnstructuredPowerPointLoader(temp_filepath)
                    
                    docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            vector_store.add_documents(splits)
            st.success("Documents successfully added to the long-term memory!")

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- NEW FEATURE: Create a new chain that considers conversation history ---
history_aware_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question."),
    ("user", "{chat_history}\nFollow Up Input: {input}"),
    ("user", "Standalone question:"),
])
history_aware_retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)

# --- NEW FEATURE: Create a chain to answer the question based on context ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based only on the following context:\n\n{context}"),
    ("user", "Question: {input}"),
])
Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

# --- NEW FEATURE: Combine the chains ---
rag_chain = create_retrieval_chain(history_aware_retriever_chain, Youtube_chain)

# --- Display Chat Interface ---
# Display previous chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("Ask a question about your documents"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the new RAG chain with history
                response = rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": prompt
                })
                
                response_text = response["answer"]
                st.markdown(response_text)
                
                # --- NEW FEATURE: Display source documents in an expander ---
                with st.expander("Show Sources"):
                    for i, doc in enumerate(response["context"]):
                        st.info(f"Source {i+1}: From '{os.path.basename(doc.metadata.get('source', 'Unknown'))}'")
                        st.markdown(f"> {doc.page_content}")

            except Exception as e:
                response_text = f"An error occurred: {e}"
                st.error(response_text)
    
    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response_text))
