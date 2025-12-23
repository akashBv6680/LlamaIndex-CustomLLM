"""RAG (Retrieval-Augmented Generation) Pipeline using LlamaIndex."""
import logging
from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.response.schema import Response
from config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for custom LLMs."""

    def __init__(self):
        """Initialize RAG pipeline with LlamaIndex components."""
        self.llm = OpenAI(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
        )
        
        self.embed_model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        logger.info("RAG Pipeline initialized")

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
        """Create vector index from documents."""
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
                similarity_top_k=settings.TOP_K,
            )
            logger.info(f"Created index with {len(nodes)} nodes")
            return self.index
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def query(self, query_text: str) -> Response:
        """Execute query against the RAG pipeline."""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Load documents and create index first.")
        
        try:
            response = self.query_engine.query(query_text)
            logger.info(f"Query executed: {query_text[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error executing query: {e}")
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
