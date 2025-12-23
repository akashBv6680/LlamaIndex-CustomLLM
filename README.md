# LlamaIndex Custom LLM with Google Gemini

🚀 A production-ready Retrieval-Augmented Generation (RAG) system built with **LlamaIndex** and **Google Gemini API**.

This project demonstrates how to build custom LLMs without OpenAI, using Google's state-of-the-art Gemini models for both generation and embedding.

## Features

✨ **Core Capabilities:**
- 📚 Multi-document support (TXT, PDF, Markdown)
- 🧠 Advanced semantic search with Gemini embeddings
- 💬 Conversational AI powered by Gemini Pro
- 🔄 RAG pipeline with LlamaIndex
- 📊 Interactive Streamlit UI
- ⚙️ Configurable model parameters
- 🛡️ Safety filters for content moderation

## Technology Stack

- **Framework:** [LlamaIndex](https://www.llamaindex.ai)
- **LLM:** Google Gemini Pro
- **Embeddings:** Google Gemini Embeddings
- **UI:** Streamlit
- **Language:** Python 3.9+

## Project Structure

```
LlamaIndex-CustomLLM/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Project dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Setup Instructions

### 1. Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Create a new API key for your project
4. Copy the API key

### 2. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/akashBv6680/LlamaIndex-CustomLLM.git
cd LlamaIndex-CustomLLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Create .env file from template
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

### 4. Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
```

## Usage

### Document Management

1. **Initialize Pipeline:** Click "🔄 Initialize Pipeline" to set up RAG system
2. **Upload Documents:** Use file uploader to add TXT, PDF, or Markdown files
3. **Process Documents:** Click "📤 Process Documents" to create embeddings and index

### Query Interface

1. **Ask Questions:** Type your question about uploaded documents
2. **Get Answers:** Gemini AI retrieves relevant content and generates response
3. **View Sources:** Expand "📖 View Sources" to see referenced documents

### Statistics Dashboard

- View current model configuration
- Monitor chunk sizes and retrieval parameters
- Check embedding model details

## Key Files

### `config.py`
Centralized configuration management:
- Gemini API settings
- Model parameters (temperature, top_p, top_k)
- Chunk size and overlap
- Safety filter settings

### `rag_pipeline.py`
Core RAG implementation:
- `RAGPipeline` class for orchestration
- Document loading and indexing
- Query execution and retrieval
- Dynamic document addition

### `app.py`
Streamlit UI with:
- 4 interactive tabs
- File upload interface
- Query engine
- Statistics dashboard
- About section

## Configuration

Edit `config.py` to customize:

```python
# Model Settings
MODEL_NAME = "gemini-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Retrieval Settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20
RETRIEVAL_TOP_K = 3
```

## Advanced Usage

### Custom Safety Settings

Modify safety filters in `config.py`:

```python
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]
```

### Integration with Other Projects

```python
from rag_pipeline import RAGPipeline
from config import settings

# Initialize RAG
rag = RAGPipeline()

# Load documents
docs = rag.load_documents(["path/to/docs"])

# Create index
rag.create_index(docs)

# Query
response = rag.query("Your question here")
print(response)
```

## Deployment

### Deploy on Streamlit Cloud

1. Push code to GitHub
2. Visit [Streamlit Cloud](https://share.streamlit.io)
3. Connect GitHub repository
4. Set environment variables in Streamlit Cloud settings
5. Deploy!

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

**Issue: API Key not working**
- Verify key in Google AI Studio
- Check .env file format
- Restart Streamlit app

**Issue: Documents not loading**
- Ensure documents are in supported formats
- Check file permissions
- Verify document content is readable

**Issue: Slow queries**
- Reduce CHUNK_SIZE for smaller contexts
- Lower RETRIEVAL_TOP_K for fewer results
- Use smaller documents

## Performance Tips

- Optimal chunk size: 512-1024 tokens
- Balance top_k: 3-5 for quality results
- Temperature: 0.3-0.7 for factual responses
- Index rebuild after adding >100 documents

## Security Notes

⚠️ **Never commit `.env` file to version control**
- Always use `.env.example` as template
- Rotate API keys regularly
- Use environment variables for production

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## License

MIT License - feel free to use in your projects!

## Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [Google Gemini API](https://makersuite.google.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [RAG Systems Guide](https://docs.llamaindex.ai/en/stable/understanding_rag/)

## Support

For issues and questions:
- 📝 GitHub Issues
- 💬 Discussions
- 🐛 Bug Reports

---

Built with ❤️ using LlamaIndex and Gemini API
