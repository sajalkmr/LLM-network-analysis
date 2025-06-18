# LLM-based Network Traffic Analysis System

An advanced Intrusion Detection System (IDS) that leverages Large Language Models (LLMs) to enhance network security. This project integrates LangChain with the UNSW-NB15 dataset, using HuggingFace Embeddings and Chroma for efficient vector storage and retrieval. The system demonstrates superior threat detection capabilities compared to traditional ML models.

## Key Features

- LLM-powered network traffic analysis and intrusion detection
- LangChain integration for real-world data processing
- Vector storage using HuggingFace Embeddings and Chroma
- Baseline comparison with traditional ML models
- Real-time network traffic simulation capabilities
- Flexible deployment options (API-based or on-premise)

## Outcomes

- Improved threat detection accuracy over traditional ML models
- Enhanced contextual understanding of attack patterns
- Real-time traffic analysis capabilities
- Efficient data retrieval and processing
- Scalable deployment options for different organizations

## Prerequisites

- Python 3.8+
- Google API Key for Gemini LLM
- Required Python packages (see requirements.txt)

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/sajalkmr/LLM-network-analysis.git
cd LLM-network-analysis
pip install -r requirements.txt
```

2. Configure API Key:
   - Add to `.streamlit/secrets.toml`: `GOOGLE_API_KEY = "your-api-key-here"`
   - Or set environment variable: `GOOGLE_API_KEY`

3. Build database and run:
```bash
python3 build_chroma_db.py
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `build_chroma_db.py`: Vector database builder
- `requirements.txt`: Dependencies
- `chroma_db/`: Vector database (git-ignored)
- `.streamlit/`: Configuration and secrets

## License

MIT License 
