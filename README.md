# EU-Tax Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot designed to answer complex questions based on EU tax reporting regulations. It uses LangChain, FAISS, HuggingFace Embeddings, and OpenAI GPT models to deliver accurate, source-grounded answers from custom datasets.

## Installation  

1. Clone this repository:  
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the chatbot: (in the code folder)
   ```sh
   cd code
   streamlit run app.py
   ```
2. Run Batch Evaluation to Get Multiple Answers
   - Put the question csv in the evaluation/evaluation_question/ folder
   ```sh
   cd code
   python evaluation.py
   ```
   - The answer will show up in the evaluation/evaluation_answer/ foler.


## Features
- 🧠 RAG-based Question Answering using LangChain
- 📄 Document Parsing from CSV-based scraped content
- 🔎 FAISS Vector Store for semantic search
- 🤖 GPT-4 Chat Integration via OpenAI API
- 💬 Streamlit Web Interface
- 🗃️ Chat History with context persistence
- 📋 Evaluation Mode for batch question testing
- ✅ Test Result Logging with context/answers and timestamps