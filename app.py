"""Main Streamlit application for LlamaIndex Custom LLM."""
import streamlit as st
import logging
from pathlib import Path
from rag_pipeline import RAGPipeline
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Main interface
st.title("🚀 LlamaIndex Custom LLM Application")
st.markdown(
    """Welcome to the Custom LLM powered by LlamaIndex RAG Pipeline.
    Upload documents, create embeddings, and query your custom knowledge base."""
)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📚 Document Management", "🔍 Query Engine", "📊 Statistics", "ℹ️ About"]
)

with tab1:
    st.subheader("📁 Upload Documents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload text or PDF documents",
            type=["txt", "pdf", "md"],
            accept_multiple_files=True,
        )
    
    with col2:
        if st.button("🔄 Initialize Pipeline", use_container_width=True):
            st.session_state.rag_pipeline = RAGPipeline()
            st.success("✅ RAG Pipeline initialized!")
            st.session_state.documents_loaded = False
    
    if uploaded_files and st.session_state.rag_pipeline:
        if st.button("📤 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = Path("temp_docs")
                    temp_dir.mkdir(exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        file_path = temp_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load and process documents
                    documents = st.session_state.rag_pipeline.load_documents(
                        [str(temp_dir)]
                    )
                    st.session_state.rag_pipeline.create_index(documents)
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Processed {len(documents)} documents!")
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")
                    logger.error(f"Document processing error: {e}")

with tab2:
    st.subheader("🤖 Ask Questions")
    
    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please upload and process documents first in the Document Management tab.")
    else:
        query = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
        )
        
        if st.button("🔎 Search", use_container_width=True):
            if query.strip():
                with st.spinner("Searching..."):
                    try:
                        response = st.session_state.rag_pipeline.query(query)
                        
                        st.subheader("📝 Response:")
                        st.write(response)
                        
                        # Display source documents
                        if hasattr(response, "source_nodes") and response.source_nodes:
                            with st.expander("📖 View Sources"):
                                for i, source in enumerate(response.source_nodes, 1):
                                    st.write(f"**Source {i}:**")
                                    st.write(source.node.get_content()[:500])
                    except Exception as e:
                        st.error(f"❌ Error executing query: {e}")
                        logger.error(f"Query error: {e}")
            else:
                st.warning("Please enter a question.")

with tab3:
    st.subheader("📊 Pipeline Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model",
            settings.MODEL_NAME,
        )
    
    with col2:
        st.metric(
            "Embedding Model",
            settings.EMBEDDING_MODEL,
        )
    
    with col3:
        st.metric(
            "Temperature",
            f"{settings.TEMPERATURE}",
        )
    
    st.markdown("---")
    st.info(
        f"""**Configuration Summary:**
        - Chunk Size: {settings.CHUNK_SIZE} tokens
        - Chunk Overlap: {settings.CHUNK_OVERLAP} tokens
        - Top-K Results: {settings.TOP_K}
        - Max Tokens: {settings.MAX_TOKENS}"""
    )

with tab4:
    st.subheader("About This Application")
    st.markdown(
        f"""### {settings.APP_NAME}
        
        This application demonstrates a production-ready RAG (Retrieval-Augmented Generation)
        system built with LlamaIndex and OpenAI APIs.
        
        **Features:**
        - 📚 Multi-document support (TXT, PDF, MD)
        - 🧠 Advanced embedding with OpenAI
        - 🔍 Semantic search and retrieval
        - 💬 Conversational AI powered by GPT
        - 📊 Interactive statistics dashboard
        
        **Technologies:**
        - LlamaIndex Framework
        - OpenAI API (GPT-3.5, Embeddings)
        - Streamlit for UI
        - Python 3.9+
        
        Made with ❤️ using LlamaIndex
    """
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [LlamaIndex](https://www.llamaindex.ai) | "
    "[GitHub](https://github.com/akashBv6680/LlamaIndex-CustomLLM)"
)
