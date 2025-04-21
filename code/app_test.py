import streamlit as st
import requests

# Load secrets from .streamlit/secrets.toml
host = st.secrets["databricks"]["host"]
token = st.secrets["databricks"]["token"]
embedding_endpoint = st.secrets["databricks"]["embedding_endpoint"]
chat_endpoint = st.secrets["databricks"]["chat_endpoint"]

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Page setup
st.set_page_config(page_title="ESRS Chatbot", layout="wide")
st.title("💬 ESRS Chatbot — RAG via Databricks Endpoints")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

# Input area
question = st.text_input("Ask a question about ESRS:")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Ask") and question.strip():
        # Embed the question
        with st.spinner("🔎 Embedding your question..."):
            embed_res = requests.post(
                f"{host}/serving-endpoints/{embedding_endpoint}/invocations",
                headers=headers,
                json={"input": [question]}
            )

            if embed_res.status_code != 200:
                st.error(f"Embedding failed: {embed_res.text}")
                st.stop()

            query_embedding = embed_res.json()["data"][0]["embedding"]

        # Simulated retrieval (replace with actual vector search logic)
        retrieved_chunks = [
            "The ESRS framework is part of the EU's Corporate Sustainability Reporting Directive (CSRD).",
            "It provides a standardized method for companies to report environmental, social, and governance data."
        ]
        st.session_state.retrieved_chunks = retrieved_chunks

        # Construct the context and system message
        context = "\n\n".join(retrieved_chunks)
        system_prompt = f"""You are an expert on EU sustainability reporting standards. Use the context below to answer questions.

Context:
{context}
"""

        # Append the user question to chat history
        st.session_state.history.append({"role": "user", "content": question})

        # Construct full message list
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.history

        # Call the chat endpoint
        with st.spinner("🤖 Generating answer..."):
            chat_res = requests.post(
                f"{host}/serving-endpoints/{chat_endpoint}/invocations",
                headers=headers,
                json={"messages": messages}
            )

            if chat_res.status_code != 200:
                st.error(f"Generation failed: {chat_res.text}")
                st.stop()

            answer = chat_res.json()["choices"][0]["message"]["content"]
            st.session_state.history.append({"role": "assistant", "content": answer})

# Display conversation history
with col1:
    if st.session_state.history:
        st.markdown("### 💬 Conversation History")
        for i in range(len(st.session_state.history) - 1, -1, -2):
            user_msg = st.session_state.history[i - 1] if i - 1 >= 0 else None
            bot_msg = st.session_state.history[i]
            if user_msg:
                st.markdown(f"**🧑 You:** {user_msg['content']}")
            st.markdown(f"**🤖 Assistant:** {bot_msg['content']}")

# Display retrieved chunks for debugging / transparency
with col2:
    st.markdown("### 📄 Retrieved Context")
    if st.session_state.retrieved_chunks:
        for i, chunk in enumerate(st.session_state.retrieved_chunks):
            st.markdown(f"**Chunk {i+1}:** {chunk}")
    else:
        st.info("No retrieved context yet.")

# Optional: Clear chat history
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.retrieved_chunks = []
        st.experimental_rerun()