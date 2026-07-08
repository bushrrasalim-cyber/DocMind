import hashlib
import streamlit as st
import os
st.set_page_config(
page_title="DocMind",
page_icon= "📄",
layout="wide",
initial_sidebar_state="expanded")

st.markdown("""
<style>
/*Hide Streamlit menu and footer*/
#MainMenu{visibility: hidden;}
footer {visibility: hidden;}
#header{visibility:hidden;}
            
</style>
""",unsafe_allow_html = True)
    
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader ,StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
# import the correct embeddings class
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader 

llm = OpenAI(
    model="gpt-4.1-mini",
    api_key= st.secrets["OPENAI_API_KEY"]
    
    )
embed_model = OpenAIEmbedding(
    api_key=st.secrets["OPENAI_API_KEY"]
)
Settings.llm = llm
Settings.embed_model = embed_model
os.makedirs("storage", exist_ok=True)

def load_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    reader = PDFReader()
    documents = reader.load_data(file="temp.pdf")
    return documents

def create_index(documents,pdf_hash):
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(
        persist_dir = f"storage/{pdf_hash}"
    )
    return index


def create_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=10
    )


def get_response(query, query_engine):
    response = query_engine.query(query)

    print("\n" + "=" * 60)
    print("Retrieved Chunks:", len(response.source_nodes))

    for i, node in enumerate(response.source_nodes):
        print(f"\n----- Chunk {i+1} -----")
        print(node.text[:800])
    return response


def get_pdf_hash(uploaded_file): 
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()


def main():
    st.title("📄DocMind")
    st.caption("chat with your documents")
    st.divider()

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "messages" not in st.session_state:
        st.session_state.messages = []


    if st.sidebar.button("🗑 Reset Document",use_container_width = True):
        st.session_state.pop("vector_index", None)
        st.session_state.pop("query_engine", None)
        st.session_state.pop("file_names", None)
        st.session_state.pop("messages", None)
        st.session_state.pop("loaded", None)
        st.session_state.uploader_key += 1
        st.rerun()
    

    st.sidebar.header("📂Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload a PDF",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    col1 , col2 = st.columns(2)
    with col1:
        if uploaded_files:
            st.metric("📄 Documents", len(uploaded_files)) 
        else: 
            st.metric("📄 Documents", 0)
    with col2:
        st.metric("💬 Messages", len(st.session_state.messages))
    st.sidebar.divider()
    if uploaded_files:
        st.sidebar.success(f"✅{len(uploaded_files)} PDF(s) Loaded ")
        
        for files in uploaded_files:
            st.sidebar.write(f"📄 {files.name}")
    if uploaded_files:
        file_names = [file.name for file in uploaded_files]

        if (
            "vector_index" not in st.session_state
            or "file_names" not in st.session_state
            or file_names != st.session_state.file_names
        ):
            
            pdf_hash = get_pdf_hash(uploaded_files[0])
            storage_path = f"storage/{pdf_hash}"
            if os.path.exists(storage_path):
                storage_context = StorageContext.from_defaults(
                    persist_dir=storage_path)
                index = load_index_from_storage(storage_context)
            else:

                 all_documents = []

                 with st.spinner("Processing PDF..."):
                    for uploaded_file in uploaded_files:
                        documents = load_pdf(uploaded_file)
                        all_documents.extend(documents)

                    index = create_index(all_documents, pdf_hash)
           
            st.session_state.vector_index = index
            st.session_state.query_engine = create_query_engine(index)
            st.session_state.file_names = file_names

        if "loaded" not in st.session_state:
            st.success("PDF processed successfully")
            st.session_state.loaded = True
        query_engine = st.session_state.query_engine
        chat_container = st.container(border = True)
        with chat_container:

        
            if not st.session_state.messages:
                st.info("""
        👋 Welcome to document gpt

        Upload one or more PDF or ask questions like
            - 📄 Summarize this document
            - 🔍 Explain Chapter 3
            - 📌 List the important points
            - ⚖ Compare two PDFs""")
            for messages in st.session_state.messages:        
                with st.chat_message(messages["role"]):
                    st.write(messages["content"])
            query = st.chat_input("💬 Ask anything about your document...:")

        if query:
            st.session_state.messages.append(
            {"role": "user", "content": query}
            )
            response = get_response(query, query_engine)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.response}
            )
            st.rerun()


if __name__ == "__main__":
    main()