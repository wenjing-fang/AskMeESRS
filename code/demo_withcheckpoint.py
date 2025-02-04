import os
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import streamlit as st
from langgraph.graph import MessagesState
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import csv
import time


print('Setting up...')
# Set up the LLM and embeddings
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vector_store = InMemoryVectorStore(embeddings)

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))



# Define folder path for PDFs
folder_path = "../dataset"


# Function to extract text from PDFs

def extract_text_from_pdfs(pdf_files):
    filename = "../checkpoints/all_text.txt"

    # Function to save all_text to a file
    def save_all_text(all_text, filename=filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"Text saved to {filename} in plain text format")

    # Check if the file already exists
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            all_text = f.read()
        print("All text loaded from file")
    else:
        all_text = ""
        total_files = len(pdf_files)
        print(f"Starting text extraction from {total_files} PDFs...")

        for i, file_name in enumerate(pdf_files):
            print(f"Processing file {i+1}/{total_files}: {file_name}")  # Informing the user which file is being processed
            with pdfplumber.open(file_name) as pdf:
                for page in pdf.pages:
                    all_text += page.extract_text()
            
            # Show progress as a percentage
            progress = ((i + 1) / total_files) * 100
            print(f"Progress: {progress:.2f}%")

        save_all_text(all_text)
        print("Text extraction complete!")

    return all_text

def create_vector_store(text):
    filename = "../checkpoints/faiss_index"
        
    if not os.path.exists(filename):
        print("Creating a new vector store...")
        
        # Initialize the FAISS vector store
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents([text])
        
        # Track chunking progress
        total_docs = len(documents)
        print(f"Splitting text into {total_docs} chunks...")

        for i, doc in enumerate(documents):
            vector_store.add_documents([doc])
            
            # Show progress in terminal
            progress = ((i + 1) / total_docs) * 100
            print(f"Progress: {progress:.2f}% - Processed chunk {i + 1}/{total_docs}")
        
        # Save the vector store locally
        vector_store.save_local(filename)
        save_documents_to_csv(vector_store)
        print(f"Vector store saved to {filename}")

    else:
        # Load the existing vector store
        print(f"Loading existing vector store from {filename}...")
        vector_store = FAISS.load_local(
            filename, embeddings, allow_dangerous_deserialization=True
        )

    print("Text chunking and vector store creation completed!")
    return vector_store

import csv
def save_documents_to_csv(vector_store, filename="../checkpoints/vector_store_documents.csv"):
    print("Exporting documents to CSV...")
    
    # Open a CSV file in write mode
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(["Document ID", "Content Preview", "Full Content"])
        
        # Retrieve and write each document
        for doc_id, document in vector_store.docstore._dict.items():
            content_preview = document.page_content[:100] + "..."  # Only preview first 100 characters
            csv_writer.writerow([doc_id, content_preview, document.page_content])
    
    print(f"Documents successfully saved to {filename}")

# Tool for retrieval
@tool(response_format="content_and_artifact")
def retrieve(query: str,csv_file: str = "../checkpoints/retrieved_docs.csv"):
    """Retrieve information related to a query."""
    print('Retrieving...')
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    with open(csv_file, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if file.tell() == 0:
            writer.writerow(['Query',"ID", "Content"])
        for doc in retrieved_docs:
            writer.writerow([query,doc.id, doc.page_content])
    
    print('Logged to CSV.')
    
        
    print(f'Retrieved {len(retrieved_docs)} documents.')
    return serialized, retrieved_docs

# # Define LangGraph setup
# def setup_langgraph():
#     graph_builder = StateGraph(MessagesState)
    
#     def query_or_respond(state: MessagesState):
#         llm_with_tools = llm.bind_tools([retrieve])
#         response = llm_with_tools.invoke(state["messages"])
#         return {"messages": [response]}
    
#     def generate_answer(state: MessagesState):
#         recent_tool_messages = []
#         for message in reversed(state["messages"]):
#             if message.type == "tool":
#                 recent_tool_messages.append(message)
#             else:
#                 break
#         tool_messages = recent_tool_messages[::-1]
#         docs_content = "\n\n".join(doc.content for doc in tool_messages)
#         system_message_content = (
#             "You are an assistant for question-answering tasks. "
#             "Use the following pieces of retrieved context to answer "
#             "the question. If you don't know the answer, say that you "
#             "don't know. Use three sentences maximum and keep the "
#             "answer concise."
#             "\n\n"
#             f"{docs_content}"
#         )
#         conversation_messages = [
#             message
#             for message in state["messages"]
#             if message.type in ("human", "system")
#             or (message.type == "ai" and not message.tool_calls)
#         ]
#         prompt = [SystemMessage(system_message_content)] + conversation_messages
#         response = llm.invoke(prompt)
#         return {"messages": [response]}
    
#     graph_builder.add_node(query_or_respond)
#     graph_builder.add_node(generate_answer)
#     graph_builder.add_edge("query_or_respond", "generate_answer")
#     graph_builder.set_entry_point("query_or_respond")
#     return graph_builder.compile(checkpointer=MemorySaver())

# graph = setup_langgraph()




agent_executor = create_react_agent(llm, [retrieve], checkpointer=MemorySaver())


# Process pre-existing documents
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
if pdf_files:
    text = extract_text_from_pdfs(pdf_files)
    vector_store = create_vector_store(text)
else:
    raise FileNotFoundError(f"No PDFs found in {folder_path}. Ensure the folder contains documents.")


# Streamlit App
st.title("ESRS Document Assistant")
st.write("Ask your ESRS-related questions!")

# Session state management
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Chat interface
st.write("### Chat with the Assistant")
input_message = st.text_input("Enter your question:")


# Modify the part where the question is asked to include the logging mechanism
# Function to handle the button action and log interactions
if st.button("Ask") and input_message.strip():
    try:
        response_message = ""
        
        # Stream the response from the agent
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config={"configurable": {"thread_id": "def234"}},
        ):
            response_message = event["messages"][-1].content

        
        # Append the conversation to the session
        st.session_state["conversation"].append({"user": input_message, "agent": response_message})
        
        # Log the interaction into a CSV file
        # func.log_interaction_to_csv(event)  # Pass the entire event for logging
        
    except Exception as e:
        st.error(f"Error: {e}")


# Display conversation history
for message in reversed(st.session_state["conversation"]):
    st.write(f"**You:** {message['user']}")
    st.write(f"**Assistant:** {message['agent']}")