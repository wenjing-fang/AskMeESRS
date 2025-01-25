# Demo_1

This project implements a conversational AI agent that answers questions related to the materiality assessment process using a custom-built retrieval-augmented generation (RAG) pipeline.

## Features

- **Chat Model**: Using Ollama that runs locally.
- **Document Loader**: Automatically loads web content from a specified URL.
- **Text Splitting**: Splits large content into manageable chunks for efficient processing.
- **Vector Search**: Utilizes embeddings for similarity-based document retrieval.
- **Conversational Agent**: A chat interface powered by a Large Language Model (LLM) to answer user queries based on the retrieved context.
- **Streamlit Web App**: A user-friendly interface to interact with the chatbot.

## Requirements

- Python 3.8 or higher
- ollama installed locally https://ollama.com/
- Libraries listed in `requirements.txt`

## Installation and Usage

1. Clone the repository:

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Setup Chatmodel Ollama:
    ```bash
    ollama serve
    ollama pull llama3.1

4. Setup LANGSMITH environment variable:
    ```bash
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=your-langsmith-api-key

5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    Open the URL provided by Streamlit (usually http://localhost:8501) in your web browser.


