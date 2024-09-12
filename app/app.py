import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


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
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
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
    # Note: We don't use StreamingStdOutCallbackHandler here anymore
    llm = Ollama(model="gemma2:2b", temperature=0.4)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    # Enhanced system prompt
    template = """Du bist ein hilfreicher Assistent für das Tesla-Handbuch. Deine Aufgabe ist es, Fragen über Tesla-Fahrzeuge basierend auf dem offiziellen Handbuch zu beantworten.

    Befolge diese Richtlinien:
    1. Antworte nur mit Informationen aus dem Handbuch. Wenn du unsicher bist, sage das ehrlich.
    2. Gib genaue Zitate aus dem Handbuch an, wenn möglich. Verwende dafür Anführungszeichen.
    3. Verweise auf spezifische Abschnitte oder Seitenzahlen, wenn verfügbar.
    4. Sei präzise und knapp in deinen Antworten.
    5. Wenn eine Frage nicht mit dem Handbuch zusammenhängt, weise höflich darauf hin.

    Kontext: {context}
    Chat-Verlauf: {chat_history}
    Menschliche Frage: {question}
    Assistenten-Antwort:"""

    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"], template=template
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        _vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return qa_chain


# Aufgabe 1: ChatUI
def main():
    st.set_page_config(
        page_title="Tesla Handbuch Chatbot", page_icon="🚗", layout="wide"
    )
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
            stream_handler = StreamHandler(response_container)

            response = qa_chain({"question": prompt}, callbacks=[stream_handler])

            # Ensure the final response is displayed without the cursor
            response_container.markdown(stream_handler.text)

        st.session_state.messages.append(
            {"role": "assistant", "content": stream_handler.text}
        )

        # Anzeigen der Quellen
        # if show_sources and "source_documents" in response:
        with st.expander("Quellen", expanded=True):
            for i, doc in enumerate(response["source_documents"], 1):
                st.info(f"Seite: {doc.metadata.get('page', 'N/A')}")
                st.write(doc.page_content)
                st.markdown("---")


if __name__ == "__main__":
    main()
