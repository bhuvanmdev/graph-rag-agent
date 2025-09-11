# 🎫 Customer Support Copilot - RAG System

## 📋 What is This Project?

The **Customer Support Copilot** is an intelligent AI-powered system designed to revolutionize customer support workflows for data catalog platforms like Atlan. It combines advanced machine learning techniques with a user-friendly web interface to automate ticket classification, provide instant responses using retrieval-augmented generation (RAG), and intelligently route complex issues to human agents.

### 🎯 Core Purpose
- **Automate Routine Support**: Handle common customer inquiries automatically using AI
- **Improve Response Quality**: Provide accurate, context-aware answers based on official documentation
- **Reduce Resolution Time**: Classify tickets instantly and route appropriately
- **Scale Support Operations**: Process multiple tickets simultaneously with consistent quality

## 🏗️ Architecture Overview

### Data Flow Pipeline
1. **Knowledge Base Creation**: Web scraping → Content processing → Vector embeddings → Neo4j storage
2. **Ticket Processing**: Classification → Routing decision → RAG response generation
3. **Human Oversight**: Quality monitoring → Escalation handling → Continuous learning

### Technical Stack
- **Frontend**: Gradio web interface with multi-tab dashboard
- **Backend**: Python FastAPI-style processing with async capabilities
- **Database**: Neo4j graph database for knowledge base + SQLite for ticket management
- **AI/ML**: Google Gemini LLM + Sentence Transformers for embeddings
- **Web Scraping**: crawl4ai for robust website content extraction

## 🚀 Key Features

### 🤖 Intelligent Ticket Classification
- **Multi-dimensional Analysis**: Topic, sentiment, and priority classification
- **Real-time Processing**: Instant classification results with confidence scores
- **Adaptive Learning**: Continuous improvement through feedback loops

### 📚 Retrieval-Augmented Generation (RAG)
- **Contextual Responses**: Answers based on official documentation and knowledge base
- **Source Attribution**: Transparent citation of information sources
- **Relevance Scoring**: Optimized retrieval using vector similarity search

### 🎨 Modern Web Interface (Gradio)
- **Multi-Tab Dashboard**: Organized workflow for different user roles
- **Real-time Updates**: Live processing status and results
- **Interactive Elements**: Dynamic forms, tables, and visualization
- **Responsive Design**: Works across desktop and mobile devices

### 🗄️ Dual Database Architecture
- **Neo4j Graph Database**: Knowledge base with vector indexing and graph relationships
- **SQLite Database**: Ticket management with ACID compliance and local storage

## 📋 Prerequisites

### System Requirements
- **Python**: 3.13+ (recommended for optimal performance)
- **Memory**: 8GB+ RAM (16GB recommended for large knowledge bases)
- **Storage**: 10GB+ free space for databases and embeddings
- **Network**: Stable internet connection for API calls and web scraping

### External Dependencies
1. **Neo4j Database** (version 5.11+ for vector indexing)
   - Download from: https://neo4j.com/download/
   - Enable vector index capabilities
   - Configure authentication and network access

2. **Google Gemini API Key**
   - Sign up at: https://makersuite.google.com/app/apikey
   - Enable Gemini 1.5 Flash model
   - Monitor usage and billing

## 🛠️ Installation & Setup

### 1. Clone and Environment Setup
```bash
git clone <your-repo-url>
cd customer-support-copilot
```

### 2. Python Environment
```bash
# Using uv (recommended - faster package management)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Neo4j Database Configuration
```bash
# Start Neo4j Desktop or Server
# Create a new database project
# Note the connection details:
# - URI: bolt://localhost:7687 (default)
# - Username: neo4j (default)
# - Password: your_chosen_password
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Optional: Custom Settings
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=1000
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 📚 Usage Guide

### Phase 1: Knowledge Base Setup

#### Data Ingestion Pipeline
```bash
# Basic usage - scrape documentation site
python ingest_pipeline.py --root-url "https://docs.atlan.com" --max-pages 50

# Advanced options
python ingest_pipeline.py \
  --root-url "https://docs.atlan.com" \
  --max-pages 100 \
  --clear-db \
  --verbose
```

**Ingestion Parameters:**
- `--root-url`: Starting URL for web scraping (required)
- `--max-pages`: Maximum pages to scrape (default: 10)
- `--clear-db`: Clear existing database before ingestion
- `--verbose`: Enable detailed logging

**What happens during ingestion:**
1. **Web Crawling**: BFS traversal of website links
2. **Content Extraction**: HTML to Markdown conversion
3. **Text Chunking**: Semantic splitting preserving structure
4. **Vector Generation**: Sentence transformer embeddings
5. **Graph Storage**: Neo4j nodes and relationships creation

### Phase 2: System Operation

#### Launch the Web Interface
```bash
python rag_app.py
```

The system will start at: `http://localhost:7860`

## 🎨 Gradio Web Interface Deep Dive

### Interface Architecture
The Gradio interface is built with a **tabbed architecture** designed for different user workflows:

#### 1. 📊 Bulk Ticket Classification Dashboard
**Purpose**: Process multiple tickets simultaneously for batch operations

**Key Features:**
- **Sample Ticket Library**: Pre-loaded realistic customer support tickets
- **Bulk Processing Engine**: Parallel classification and response generation
- **Results Visualization**: Color-coded tables with priority indicators
- **Performance Metrics**: Processing time and success rates

**Use Cases:**
- Demo presentations
- Batch ticket analysis
- Performance benchmarking
- Quality assurance testing

#### 2. 🤖 Interactive AI Agent
**Purpose**: Real-time ticket processing with human-in-the-loop capabilities

**Workflow:**
1. **Ticket Submission**: Customer enters subject and description
2. **Backend Analysis**: AI classification (topic, sentiment, priority)
3. **Routing Decision**: Automatic RAG vs human assignment
4. **Response Generation**: Instant AI response for suitable topics
5. **Human Escalation**: One-click human review request

**Advanced Features:**
- **Multi-view Output**: Separate backend/frontend perspectives
- **Source Transparency**: Documentation sources cited
- **Confidence Scoring**: AI certainty indicators
- **Feedback Loop**: Human review integration

#### 3. 🗄️ Ticket SQL Database
**Purpose**: Complete database management and inspection interface

**Database Tables:**
- **Tickets**: Raw ticket data with metadata
- **Classifications**: AI analysis results
- **RAG Answers**: Generated responses with sources
- **Human Reviews**: Escalation tracking and status

**Features:**
- **Real-time Updates**: Live database synchronization
- **Advanced Filtering**: Search and sort capabilities
- **Export Options**: Data export for analysis
- **Audit Trail**: Complete processing history

#### 4. 📚 Knowledge Base Query
**Purpose**: Direct access to the RAG system for documentation queries

**Capabilities:**
- **Natural Language Queries**: Conversational question answering
- **Relevance Tuning**: Adjustable retrieval parameters
- **Source Verification**: Original documentation links
- **Debug Information**: Retrieval statistics and scores

### Gradio Technical Implementation

#### CSS Styling & UX
```css
/* Custom styling for professional appearance */
.info-box {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #43e97b 100%);
  border-radius: 15px;
  box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
}

.results-table {
  max-height: 600px;
  overflow-y: auto;
  border-radius: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
```

#### State Management
- **Reactive Updates**: Real-time interface updates
- **Session Persistence**: Maintain context across interactions
- **Error Handling**: Graceful failure recovery
- **Loading States**: Progress indicators for long operations

#### Performance Optimizations
- **Async Processing**: Non-blocking operations
- **Caching**: Response caching for repeated queries
- **Pagination**: Large dataset handling
- **Lazy Loading**: On-demand content loading

## 🗄️ Neo4j Graph Database Architecture

### Why Neo4j for RAG?
Neo4j was chosen for several critical reasons:

1. **Native Vector Support**: Built-in vector indexing (Neo4j 5.11+)
2. **Graph Relationships**: Natural representation of web page links
3. **ACID Compliance**: Data integrity for production use
4. **Cypher Query Language**: Expressive graph traversals
5. **Scalability**: Handles large knowledge graphs efficiently

### Database Schema Design

#### Node Types
```cypher
// Page nodes - represent web pages
CREATE (p:Page {
  url: "https://docs.atlan.com/connectors",
  title: "Data Connectors",
  content: "# Data Connectors\n...",
  metadata: {scraped_at: datetime(), status: "processed"}
})

// Chunk nodes - represent content segments
CREATE (c:Chunk {
  text: "Atlan supports 50+ data connectors...",
  embedding: [0.1, 0.2, 0.3, ...],
  index: 0,
  chunk_type: "markdown_header"
})
```

#### Relationship Types
```cypher
// Content chunking relationship
(page:Page)-[:HAS_CHUNK]->(chunk:Chunk)

// Website link structure
(page1:Page)-[:LINKS_TO]->(page2:Page)
```

#### Vector Index Configuration
```cypher
// Create vector index for similarity search
CREATE VECTOR INDEX chunk_embedding_index
FOR (c:Chunk)
ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
```

### Query Patterns

#### Similarity Search
```cypher
// Find most relevant chunks for a query
MATCH (c:Chunk)
WHERE vector.similarity.cosine(c.embedding, $query_embedding) > 0.7
RETURN c.text, c.url, vector.similarity.cosine(c.embedding, $query_embedding) as score
ORDER BY score DESC
LIMIT 5
```

#### Contextual Retrieval
```cypher
// Get chunks with page context
MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
WHERE vector.similarity.cosine(c.embedding, $query_embedding) > 0.7
RETURN p.title, p.url, c.text, c.index
ORDER BY vector.similarity.cosine(c.embedding, $query_embedding) DESC
```

### Performance Characteristics
- **Indexing**: Sub-second vector similarity search
- **Storage**: Efficient compression for embeddings
- **Query Speed**: Millisecond response times
- **Scalability**: Handles millions of nodes and relationships

## 🤖 AI/ML Pipeline Details

### Ticket Classification System
**Input**: Raw ticket text (subject + body)
**Output**: Structured classification results

#### Classification Dimensions
1. **Topic Classification**
   - Categories: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data
   - Method: Few-shot prompting with Gemini
   - Confidence: Probability scores for each category

2. **Sentiment Analysis**
   - Scale: Positive, Neutral, Negative
   - Method: Emotion detection and polarity analysis
   - Use: Routing and priority decisions

3. **Priority Assessment**
   - Levels: P0 (Critical), P1 (High), P2 (Medium), P3 (Low)
   - Factors: Urgency keywords, business impact, SLA requirements

#### Routing Logic
```python
def should_use_rag(classification_result):
    """Determine if ticket should use RAG or human routing"""
    rag_topics = ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']

    if classification_result.topic in rag_topics:
        if classification_result.sentiment != 'Negative':
            return True

    return False  # Route to human agent
```

### RAG Response Generation

#### Retrieval Strategy
1. **Query Embedding**: Convert question to vector using same model as ingestion
2. **Similarity Search**: Find top-k most relevant chunks in Neo4j
3. **Re-ranking**: Optional re-ranking based on relevance scores
4. **Context Assembly**: Combine retrieved chunks with query

#### Generation Process
```python
def generate_rag_response(query, context_chunks):
    """Generate response using retrieved context"""

    prompt = f"""
    Based on the following documentation context, answer the user's question.
    If the context doesn't contain enough information, say so.

    Context:
    {context_chunks}

    Question: {query}

    Answer:
    """

    response = gemini.generate(prompt)
    return response
```

#### Quality Assurance
- **Source Attribution**: Always cite source documents
- **Confidence Scoring**: Indicate certainty levels
- **Fallback Handling**: Graceful degradation for low-confidence responses

## 🔧 Configuration & Optimization

### Embedding Model Selection
```python
# Recommended configurations
EMBEDDING_CONFIGS = {
    'fast': {
        'model': 'all-MiniLM-L6-v2',
        'dimensions': 384,
        'speed': 'fast',
        'quality': 'good'
    },
    'balanced': {
        'model': 'all-mpnet-base-v2',
        'dimensions': 768,
        'speed': 'medium',
        'quality': 'excellent'
    }
}
```

### Chunking Strategy Optimization
```python
# Markdown-aware chunking for better semantic preservation
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
)

# Fallback character-based splitting
character_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
```

### Performance Tuning
- **Batch Processing**: Process multiple tickets simultaneously
- **Caching**: Cache frequent queries and embeddings
- **Async Operations**: Non-blocking I/O for better responsiveness
- **Memory Management**: Streaming for large datasets

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### Neo4j Connection Problems
```bash
# Check Neo4j status
curl http://localhost:7474/

# Verify credentials
python -c "from neo4j import GraphDatabase; GraphDatabase.driver(uri, auth=(user, password)).verify_connectivity()"

# Check vector index
MATCH ()-[r]-() RETURN count(r) as relationships
```

#### Gemini API Issues
```python
# Test API connectivity
import google.generativeai as genai
genai.configure(api_key='your_key')
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Test')
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Adjust chunk size
export MAX_CHUNK_SIZE=500
export BATCH_SIZE=10
```

#### Web Scraping Failures
```python
# Test scraping
from crawl4ai import AsyncWebCrawler
async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url='https://example.com')
    print(f"Success: {result.success}")
```

## 📊 Monitoring & Analytics

### Key Metrics to Track
- **Response Accuracy**: Human feedback on AI responses
- **Processing Speed**: Average classification and response times
- **User Satisfaction**: Ticket resolution ratings
- **System Uptime**: Service availability and reliability

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('support_copilot.log'),
        logging.StreamHandler()
    ]
)
```

## 🚀 Deployment & Scaling

### Development Environment
```bash
# Local development
python rag_app.py --debug --port 7860
```

### Production Deployment
```bash
# Using Docker
docker build -t support-copilot .
docker run -p 7860:7860 -e GEMINI_API_KEY=... support-copilot

# Using Kubernetes
kubectl apply -f k8s/deployment.yaml
```

### Scaling Considerations
- **Horizontal Scaling**: Multiple instances behind load balancer
- **Database Clustering**: Neo4j cluster for high availability
- **Caching Layer**: Redis for session and response caching
- **Monitoring**: Prometheus + Grafana for observability

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-capability`
3. Make changes with tests
4. Run quality checks: `python -m pytest`
5. Submit pull request

### Code Quality Standards
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests
- **Linting**: Black formatting and flake8 compliance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini**: Advanced language model capabilities
- **Neo4j**: Powerful graph database platform
- **Gradio**: Excellent web interface framework
- **Sentence Transformers**: High-quality embedding models
- **crawl4ai**: Robust web scraping library
- **Atlan Community**: Inspiration and use case validation

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community**: Discussion forums and user groups

---

**Built with ❤️ for the data catalog community**
