import streamlit as st
from openai import OpenAI
import os
import tempfile

# Import LangChain components for the Agent
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")
st.title("🤖 AI Research Assistant")

# --- Configuration ---
with st.sidebar:
    st.header("Configuration")
    try:
        # Load secrets for deployment
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
        st.success("API keys loaded!")
    except:
        # Fallback for local testing
        OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
        TAVILY_API_KEY = st.text_input("Enter your Tavily API Key:", type="password")

    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, PPTX)",
        type=['pdf', 'docx', 'pptx'],
        accept_multiple_files=True
    )

# --- Main App Logic ---
if not all([OPENAI_API_KEY, TAVILY_API_KEY]):
    st.warning("Please provide all required API keys in the sidebar.")
    st.stop()

# Initialize models and embeddings
try:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI components: {e}")
    st.stop()
    
# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

# Process documents and create the document search tool
if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
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
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.success("Documents processed! The agent is now ready.")

# --- AGENT AND TOOLS SETUP ---
# Define the tools the agent can use
internet_search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
tools = [internet_search_tool]

# Create the document retriever tool only if documents have been processed
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever()
    document_retriever_tool = create_retriever_tool(
        retriever,
        "document_search",
        "Search for information within the user's uploaded documents. Use this for specific questions about the provided files."
    )
    tools.append(document_retriever_tool)

# Get the prompt for the agent
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent and the executor that runs it
agent = create_openai_tools_agent(llm, tools, prompt)
st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- CHAT INTERFACE ---
# Display previous chat messages
for message in st.session_state.chat_history:
    # Check the type of the message to determine the role
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else: # Assumes the other type is AIMessage
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Accept user input
if user_prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent with the user's question and chat history
                response = st.session_state.agent_executor.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_prompt
                })
                response_text = response["output"]
                st.markdown(response_text)
            except Exception as e:
                response_text = f"An error occurred: {e}"
                st.error(response_text)
    
    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    st.session_state.chat_history.append(AIMessage(content=response_text))
