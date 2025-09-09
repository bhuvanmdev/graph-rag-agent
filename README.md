# RAG Knowledge Base System

A complete Retrieval-Augmented Generation (RAG) system that scrapes websites, stores content in a Neo4j graph database, and provides a Gradio web interface for querying the knowledge base using Google Gemini LLM.

## 🚀 Features

- **Web Scraping**: Recursively crawls websites using BFS algorithm with `crawl4ai`
- **Graph Database**: Stores content and relationships in Neo4j with vector indexing
- **Smart Chunking**: Uses Markdown-aware text splitting for better semantic chunks
- **Vector Search**: Efficient similarity search using sentence-transformers embeddings
- **LLM Integration**: Google Gemini for high-quality answer generation
- **Web Interface**: User-friendly Gradio interface for querying

## 📋 Prerequisites

1. **Python 3.13+**
2. **Neo4j Database** (version 5.11+ for vector indexing)
3. **Google Gemini API Key**

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd atlan
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv (faster)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. Setup Neo4j database:
   - Install Neo4j Desktop or use Neo4j Cloud
   - Create a new database
   - Note the connection URI, username, and password

4. Create a `.env` file with your configuration:
```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (defaults shown)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## 📚 Usage

### Step 1: Data Ingestion

Run the ingestion pipeline to scrape and store website content:

```bash
python ingest_pipeline.py --root-url "https://example.com" --max-pages 20
```

**Options:**
- `--root-url`: Starting URL for scraping (default: https://gemini.google.com/faq)
- `--max-pages`: Maximum number of pages to scrape (default: 10)
- `--clear-db`: Clear existing database before ingestion

**Example:**
```bash
# Scrape documentation site
python ingest_pipeline.py --root-url "https://docs.python.org" --max-pages 50

# Clear database and scrape fresh content
python ingest_pipeline.py --root-url "https://example.com" --max-pages 30 --clear-db
```

### Step 2: Query Interface

Launch the web interface for querying:

```bash
python rag_app.py
```

This will start a Gradio web interface at `http://localhost:7860` where you can:
- Ask questions about the scraped content
- Adjust the number of chunks to retrieve
- View sources and debug information
- See database statistics

## 🏗️ Architecture

### Data Flow

1. **Ingestion Pipeline** (`ingest_pipeline.py`):
   - BFS web crawling with `crawl4ai`
   - HTML to Markdown conversion
   - Content chunking with semantic splitting
   - Vector embedding generation
   - Neo4j graph storage with relationships

2. **Query Application** (`rag_app.py`):
   - User query embedding
   - Vector similarity search in Neo4j
   - Context assembly from retrieved chunks
   - Gemini LLM answer generation
   - Gradio web interface

### Database Schema

**Nodes:**
- `Page`: Web pages with URL, title, content, metadata
- `Chunk`: Content chunks with text, embeddings, indices

**Relationships:**
- `HAS_CHUNK`: Page → Chunk (content chunking)
- `LINKS_TO`: Page → Page (website link structure)

**Indexes:**
- Vector index on `Chunk.embedding` for similarity search
- Unique constraint on `Page.url`

## 🔧 Configuration

### Embedding Model
- **Default**: `all-MiniLM-L6-v2` (384 dimensions)
- **Rationale**: Good balance of quality/speed, suitable for RAG applications
- **Consistency**: Same model used in both ingestion and query phases

### Chunking Strategy
- **Primary**: Markdown header-based splitting
- **Fallback**: Recursive character splitting (500 chars, 50 overlap)
- **Benefits**: Preserves semantic structure while maintaining context

### LLM Configuration
- **Model**: Gemini 1.5 Flash (fast responses)
- **Temperature**: 0.7 (balanced creativity/accuracy)
- **Max tokens**: 1024

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Error**:
   - Check if Neo4j is running
   - Verify connection details in `.env`
   - Ensure Neo4j version supports vector indexing (5.11+)

2. **Gemini API Error**:
   - Verify API key is correct
   - Check API quota and billing
   - Ensure internet connectivity

3. **No Results Found**:
   - Check if data was properly ingested
   - Verify vector index was created
   - Try increasing `top_k` parameter

4. **Memory Issues**:
   - Reduce `max_pages` for ingestion
   - Decrease chunk size in text splitter
   - Monitor system resources

### Debugging

Enable debug mode in the web interface to see:
- Number of chunks retrieved
- Relevance scores
- Processing time
- Error details

## 📊 Performance Tips

1. **Ingestion**:
   - Start with smaller `max_pages` for testing
   - Use `--clear-db` only when necessary
   - Monitor Neo4j memory usage

2. **Querying**:
   - Adjust `top_k` based on query complexity
   - Use specific questions for better results
   - Check source quality in results

3. **Scaling**:
   - Use Neo4j clustering for large datasets
   - Consider caching for frequent queries
   - Monitor embedding model performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **crawl4ai**: Web scraping capabilities
- **Neo4j**: Graph database and vector search
- **Sentence Transformers**: High-quality embeddings
- **Google Gemini**: Advanced language model
- **Gradio**: User-friendly web interface
