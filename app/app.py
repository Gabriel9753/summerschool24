import os
import random
import tempfile
import webbrowser
from collections import defaultdict

import google.generativeai as genai
import plotly.graph_objects as go
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from streamlit_agraph import Config, Edge, Node, agraph

load_dotenv()

# MODEL_NAME = "gemma2:2b"
MODEL_NAME = "llama3.1:8b"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Aufgabe 2: Indexing - PDF-Verarbeitung
@st.cache_resource
def load_pdf(_file_path):
    loader = PyPDFLoader(_file_path)
    return loader.load()


# Aufgabe 2: Indexing - Chunking
@st.cache_resource
def split_text(_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(_documents)


# Aufgabe 3: Embeddings
@st.cache_resource
def create_vectorstore(_chunks):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize HuggingFaceEmbeddings with GPU support
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": device}
    )

    # Create FAISS index
    vectorstore = FAISS.from_documents(_chunks, embeddings)

    return vectorstore


# Custom callback handler for streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")


# Aufgabe 4: Retrieval Pipeline
@st.cache_resource
def init_chatbot(_vectorstore):
    # llm = Ollama(model=MODEL_NAME, temperature=0.5)
    # llm = genai.GenerativeModel("gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # Enhanced system prompt
    template = """Du bist ein hilfreicher Assistent mit zusätzlichem Wissen aus PDF-Dateien.
    Deine Aufgabe ist es, Anfragen basierend auf der PDF-Datei zu beantworten.

    Befolge diese Richtlinien:
    1. Antworte nur mit Informationen aus der PDF-Datei.
    5. Wenn eine Frage nicht mit der PDF-Datei beantwortet werden kann, gib eine entsprechende Antwort.

    Kontext: {context}
    Chat-Verlauf: {chat_history}
    Menschliche Anfrage: {question}
    Assistenten-Antwort:"""

    PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        _vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return qa_chain


def create_source_visualization(response):
    nodes = []
    edges = []
    colors = {}

    # Create a central node for the query
    query_node = Node(id="query", label="Query", size=25, color="#1f77b4", title="Central query node")
    nodes.append(query_node)

    for i, doc in enumerate(response["source_documents"]):
        doc_id = doc.metadata.get("source", f"Document {i}")
        page = doc.metadata.get("page", "N/A")

        # Assign a color to each unique document
        if doc_id not in colors:
            colors[doc_id] = f"#{random.randint(0, 0xFFFFFF):06x}"

        # Create a node for each source
        node_id = f"source_{i}"
        # Truncate the content if it's too long
        hover_text = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        node = Node(
            id=node_id, label=f"{doc_id}\nPage {page}", size=30, color=colors[doc_id], title=hover_text
        )  # This sets the hover text
        nodes.append(node)

        # Create an edge from the query to each source
        edge = Edge(source="query", target=node_id, type="SOLID")
        edges.append(edge)

    # Streamlit Agraph configuration
    config = Config(
        width=1280,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        node={"labelProperty": "label", "renderLabel": True},
        link={"labelProperty": "label", "renderLabel": True},
    )

    return nodes, edges, config


def display_sources(response):
    with st.expander("View Source Documents", expanded=False):
        # Create and display the interactive graph
        nodes, edges, config = create_source_visualization(response)
        agraph(nodes=nodes, edges=edges, config=config)

        # Add a note about hovering
        st.info("Hover over a node to view the source text. Click and drag to rearrange the graph.")


# Aufgabe 1: ChatUI
def main():
    st.set_page_config(page_title="Tesla Handbuch Chatbot", page_icon="🚗", layout="wide")
    st.title("Tesla Handbuch Chatbot 🚗")

    # Seitenleiste für Einstellungen
    # st.sidebar.title("Einstellungen")
    # show_sources = st.sidebar.checkbox("Quellen anzeigen", value=True)

    # Laden und Verarbeiten des PDFs
    # documents = load_pdf("data/genai/1902.05605.pdf")
    documents = load_pdf("data/genai/Owners_Manual_tesla.pdf")
    chunks = split_text(documents)
    vectorstore = create_vectorstore(chunks)
    qa_chain = init_chatbot(vectorstore)

    # Chat-Verlauf
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Anzeigen des Chat-Verlaufs
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Benutzereingabe
    if prompt := st.chat_input("PDF Chatbot:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            # stream_handler = StreamHandler(response_container)

            # response = qa_chain.invoke({"question": prompt}, callbacks=[stream_handler])
            response = qa_chain.invoke({"question": prompt})
            response_container.markdown(response["answer"])
            display_sources(response)

            # Ensure the final response is displayed without the cursor
            # response_container.markdown(stream_handler.text)

        # st.session_state.messages.append({"role": "assistant", "content": stream_handler.text})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

        # Anzeigen der Quellen
        # if show_sources and "source_documents" in response:


if __name__ == "__main__":
    main()
