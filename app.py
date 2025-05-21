import streamlit as st
import pandas as pd
import os
import torch
import gc

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Network Traffic Analyst",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Traffic Analysis using LLMs")
st.markdown("""
 analyze network traffic data to identify threats and provide mitigation steps.

""")

# --- Initialize Session State for Chat History ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- API Key Configuration ---
google_api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.warning("Google API Key not found. Please set GOOGLE_API_KEY as an environment variable, in .streamlit/secrets.toml, or enter it below.")
    user_input_key = st.text_input("Enter your Google API Key:", type="password")
    if user_input_key:
        os.environ["GOOGLE_API_KEY"] = user_input_key
        genai.configure(api_key=user_input_key)
    else:
        st.stop()
else:
    genai.configure(api_key=google_api_key)

# --- Define feature descriptions ---
feature_descriptions_map = {
    'srcip': 'Source IP address', 'sport': 'Source port number', 'dstip': 'Destination IP address',
    'dsport': 'Destination port number', 'proto': 'Transaction protocol',
    'state': 'Indicates to the state and its dependent protocol, e.g. ACC, CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD, URH, URN, and (-) (if not used state)',
    'dur': 'Record total duration', 'sbytes': 'Source to destination transaction bytes',
    'dbytes': 'Destination to source transaction bytes', 'sttl': 'Source to destination time to live value',
    'dttl': 'Destination to source time to live value', 'sloss': 'Source packets retransmitted or dropped',
    'dloss': 'Destination packets retransmitted or dropped', 'service': 'http, ftp, smtp, ssh, dns, ftp-data ,irc and (-) if not much used service',
    'Sload': 'Source bits per second', 'Dload': 'Destination bits per second',
    'Spkts': 'Source to destination packet count', 'Dpkts': 'Destination to source packet count',
    'swin': 'Source TCP window advertisement value', 'dwin': 'Destination TCP window advertisement value',
    'stcpb': 'Source TCP base sequence number', 'dtcpb': 'Destination TCP base sequence number',
    'smeansz': 'Mean of the flow packet size transmitted by the src',
    'dmeansz': 'Mean of the flow packet size transmitted by the dst',
    'trans_depth': 'Represents the pipelined depth into the connection of http request/response transaction',
    'res_bdy_len': 'Actual uncompressed content size of the data transferred from the server’s http service.',
    'Sjit': 'Source jitter (mSec)', 'Djit': 'Destination jitter (mSec)', 'Stime': 'record start time',
    'Ltime': 'record last time', 'Sintpkt': 'Source interpacket arrival time (mSec)',
    'Dintpkt': 'Destination interpacket arrival time (mSec)',
    'tcprtt': 'TCP connection setup round-trip time, the sum of ’synack’ and ’ackdat’.',
    'synack': 'TCP connection setup time, the time between the SYN and the SYN_ACK packets.',
    'ackdat': 'TCP connection setup time, the time between the SYN_ACK and the ACK packets.',
    'is_sm_ips_ports': 'If source (1) and destination (3)IP addresses equal and port numbers (2)(4) equal then, this variable takes value 1 else 0',
    'ct_state_ttl': 'No. for each state (6) according to specific range of values for source/destination time to live (10) (11).',
    'ct_flw_http_mthd': 'No. of flows that has methods such as Get and Post in http service.',
    'is_ftp_login': 'If the ftp session is accessed by user and password then 1 else 0.',
    'ct_ftp_cmd': 'No of flows that has a command in ftp session.',
    'ct_srv_src': 'No. of connections that contain the same service (14) and source address (1) in 100 connections according to the last time (26).',
    'ct_srv_dst': 'No. of connections that contain the same service (14) and destination address (3) in 100 connections according to the last time (26).',
    'ct_dst_ltm': 'No. of connections of the same destination address (3) in 100 connections according to the last time (26).',
    'ct_src_ltm': 'No. of connections of the same source address (1) in 100 connections according to the last time (26).',
    'ct_src_dport_ltm': 'No of connections of the same source address (1) and the destination port (4) in 100 connections according to the last time (26).',
    'ct_dst_sport_ltm': 'No of connections of the same destination address (3) and the source port (2) in 100 connections according to the last time (26).',
    'ct_dst_src_ltm': 'No of connections of the same source (1) and the destination (3) address in in 100 connections according to the last time (26).',
    'attack_cat': 'The name of each attack category. In this data set , nine categories e.g. Fuzzers, Analysis, Backdoors, DoS Exploits, Generic, Reconnaissance, Shellcode and Worms',
    'label': '0 for normal and 1 for attack records',
    'b3': 'Binary representation bit 3 of attack category (for Fuzzers, Analysis, Backdoors, DoS Exploits, Generic, Reconnaissance, Shellcode and Worms)',
    'b2': 'Binary representation bit 2 of attack category',
    'b1': 'Binary representation bit 1 of attack category',
    'b0': 'Binary representation bit 0 of attack category'
}

# --- Caching Models and Chain ---
@st.cache_resource
def load_chain_from_prebuilt_db():
    """
    Loads the pre-built Chroma DB and initializes the RAG chain.
    This function runs only once due to st.cache_resource.
    """
    st.info("Loading AI models and pre-built vector database. This will be quick!", icon="⏳")

    PERSIST_DIRECTORY = "./chroma_db"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Check if pre-built DB exists
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"Pre-built Chroma database not found at '{PERSIST_DIRECTORY}'.")
        st.error("Please run `python build_chroma_db.py` first to create the database.")
        st.stop()

    # Initialize embedding model (needed to load Chroma DB)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': 64}
    )

    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    # Load Chroma vector store from disk
    docsearch = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    st.success("Chroma vector database loaded successfully!", icon="✅")


    # Initialize the LLM
    llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17-thinking", temperature=0.1)

    # Define the prompt template
    cyber_prompt_template = """As a highly skilled cybersecurity analyst, analyze the following network traffic details:
    {context}

    Based on the provided data, answer the following question: {question}

    Provide your answer in markdown format, including:
    1.  **Threat Classification & Confidence Assessment:**
    *   **Classification:** State the most probable threat category (e.g., Normal, Exploits, DoS, Reconnaissance, Fuzzers, Generic, Shellcode, Worms, Backdoors, Analysis, or "Undetermined" if insufficient evidence).
    *   
    *   **Reasoning for Confidence:** Briefly explain *why* you are confident or not, referencing the specific data points that support your assessment.
    2.  **Confidence Level:** Provide a confidence level (High/Medium/Low) for your classification.
    **Reasoning for Confidence:** Briefly explain *why* you are confident or not, referencing the specific data points that support your assessment.
    3.  *   List specific, quantifiable features and their values from the provided records that are highly indicative of the identified classification or relevant to the question.
    *   Explain *how* these indicators (e.g., high rates, specific states, packet counts, byte transfers, unusual TCP timings, jitter, or contextual features like `ct_srv_src`) collectively point to your conclusion.
    *   Highlight any patterns, anomalies, or relationships you observe across the retrieved records.
    4.  **Actionable Mitigation Steps & Further Recommendations:**
    *   Propose concrete, actionable mitigation strategies directly related to the identified threat or observed patterns.
    *   Suggest immediate steps for incident responders or network administrators.
    *   Identify any further investigations or data points that would be crucial for a more complete understanding or deeper analysis (e.g., "Requires full packet capture for payload analysis," "Need logs from endpoint X").
    5.  
    """

    CYBER_PROMPT = PromptTemplate(
        template=cyber_prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k":75}),
        chain_type_kwargs={"prompt": CYBER_PROMPT},
        return_source_documents=True
    )

    st.success("AI models and chain initialized!", icon="✅")
    return qa_chain

# Load the QA chain (cached function)
qa_chain = load_chain_from_prebuilt_db()

# --- Main Content Area ---
st.subheader("Ask a Cybersecurity Question:")
user_query = st.text_area(
    "Enter your query about network traffic patterns or potential threats:",
    "Compare TCP vs UDP traffic patterns in these logs.",
    key="current_query_input" # Added a key for clarity
)

if st.button("Analyze Traffic", type="primary"):
    if not user_query:
        st.warning("Please enter a question to analyze.")
    else:
        with st.spinner("Analyzing... Please wait."):
            response = qa_chain({"query": user_query})

        # Store the current interaction in session state
        st.session_state.chat_history.append({
            "query": user_query,
            "result": response["result"],
            "sources": [{"content": doc.page_content, "id": doc.metadata.get('id', 'N/A')} for doc in response["source_documents"]]
        })

        st.subheader("Analysis Result:")
        st.markdown(response["result"])

        with st.expander("See Source Documents"):
            for i, doc_info in enumerate(response['source_documents']):
                # doc_info is a Document object, not a dictionary. access page_content directly.
                st.write(f"**Document {i+1} (ID: {doc_info.metadata.get('id', 'N/A')}):**")
                st.text(doc_info.page_content[:500] + "...") # Show first 500 chars
                # st.write("---")

# --- Sidebar for Chat History ---
# --- Sidebar for Chat History ---
with st.sidebar:
    st.header("Previous Analyses")

    if st.button("Clear History", key="clear_history_button"):
        st.session_state.chat_history = []
        st.experimental_rerun()  # Force immediate sidebar update

    if st.session_state.chat_history:
        # Reverse chronological display
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            query_preview = entry['query']
            if len(query_preview) > 60:
                query_preview = query_preview[:57] + "..."

            with st.expander(f"Q: {query_preview}"):
                st.markdown(f"**Question:** {entry['query']}")
                st.markdown(f"**Answer:**\n{entry['result']}")

                # List sources directly under answer, no nested expander
                if entry.get("sources"):
                    st.markdown("**Sources:**")
                    for j, source in enumerate(entry['sources']):
                        st.write(f"- Source {j+1} (ID: {source['id']}): {source['content'][:150]}...")

                st.markdown("---")  # Separator for each entry
    else:
        st.info("No previous analyses yet. Ask a question to see history here!")


st.markdown("---")
st.caption("LLM based cyber threat intelligence system")