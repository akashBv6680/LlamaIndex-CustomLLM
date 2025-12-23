"""RAG (Retrieval-Augmented Generation) Pipeline using LlamaIndex with Google Gemini."""
import logging
from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.response.schema import Response
from config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for custom LLMs using Gemini."""

    def __init__(self):
        """Initialize RAG pipeline with LlamaIndex and Gemini components."""
        self.llm = Gemini(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K,
        )
        
        self.embed_model = GeminiEmbedding(
            model_name=settings.EMBEDDING_MODEL,
        )
        
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        logger.info("RAG Pipeline initialized with Gemini")

    def load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load documents from specified paths."""
        documents = []
        for path in document_paths:
            try:
                reader = SimpleDirectoryReader(input_dir=path)
                docs = reader.load_data()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {path}")
            except Exception as e:
                logger.error(f"Error loading documents from {path}: {e}")
        return documents

    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create vector index from documents using Gemini embeddings."""
        try:
            parser = SimpleNodeParser.from_defaults(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            nodes = parser.get_nodes_from_documents(documents)
            self.index = VectorStoreIndex(
                nodes=nodes,
                embed_model=self.embed_model,
            )
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=settings.RETRIEVAL_TOP_K,
            )
            logger.info(f"Created index with {len(nodes)} nodes using Gemini")
            return self.index
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def query(self, query_text: str) -> Response:
        """Execute query against the RAG pipeline using Gemini."""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Load documents and create index first.")
        
        try:
            response = self.query_engine.query(query_text)
            logger.info(f"Query executed with Gemini: {query_text[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error executing query with Gemini: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing index."""
        if not self.index:
            raise ValueError("Index not initialized")
        
        try:
            parser = SimpleNodeParser.from_defaults(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            nodes = parser.get_nodes_from_documents(documents)
            self.index.insert_nodes(nodes)
            logger.info(f"Added {len(nodes)} new nodes to index")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
