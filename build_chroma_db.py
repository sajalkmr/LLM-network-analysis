# build_chroma_db.py
import pandas as pd
import os
import torch
import gc
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# --- Configuration ---
DATA_FILE = 'UNSW_NB15_attack_binary_bits.csv'
FEATURES_FILE = 'NUSW-NB15_features1.csv'
PERSIST_DIRECTORY = "./chroma_db"
SAMPLE_SIZE = 10000 # Keep consistent with your app for testing
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Feature Descriptions (copied from your app.py) ---
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
    'dmeansz': 'Mean of the flow packet size transmitted by the dst', # Added missing description
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

def row_to_text(row, feature_descriptions):
    return "\n".join([
        f"{feature_descriptions.get(col, col)}: {row[col]}"
        for col in row.index
        if col in feature_descriptions or col in row.index # Ensure column exists and has a description or is just the column name
    ])

def build_and_persist_chroma_db():
    print(f"--- Starting Chroma DB build process ({SAMPLE_SIZE} records) ---")

    # Load data
    try:
        df_full = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Make sure it's in the same directory.")
        return

    df_sample = df_full.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Loaded and sampled {len(df_sample)} records.")

    # Convert DataFrame rows to text documents
    print("Converting DataFrame rows to text documents...")
    texts = df_sample.apply(lambda row: row_to_text(row, feature_descriptions_map), axis=1).tolist()
    print(f"Generated {len(texts)} text documents.")

    # Initialize embedding model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device for embeddings: {device}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': 64}
    )

    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    # Ensure persist directory exists
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
        print(f"Created directory: {PERSIST_DIRECTORY}")

    # Create and persist Chroma vector store
    print(f"Creating Chroma DB and persisting to {PERSIST_DIRECTORY}...")
    docsearch = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    docsearch.persist() # Explicitly persist
    print("Chroma DB created and persisted successfully!")

if __name__ == "__main__":
    build_and_persist_chroma_db()