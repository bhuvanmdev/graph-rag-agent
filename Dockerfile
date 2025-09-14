# Graph-RAG Agent Dockerfile

FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git curl default-jre nginx netcat-traditional && \
    rm -rf /var/lib/apt/lists/*

# Download and install Neo4j Community Edition
RUN curl -fsSL https://dist.neo4j.org/neo4j-community-5.15.0-unix.tar.gz -o neo4j.tar.gz && \
    tar -xzf neo4j.tar.gz && \
    mv neo4j-community-5.15.0 /neo4j && \
    rm neo4j.tar.gz

# Set workdir
WORKDIR /app

# Clone the project
RUN git clone https://github.com/bhuvanmdev/graph-rag-agent.git .

# Create directories for Neo4j data and logs (to be bind mounted from host's current directory)
RUN mkdir -p /app/neo4j_data /app/neo4j_logs

# Set Neo4j to use these custom directories
ENV NEO4J_dbms_directories_data=/app/neo4j_data
ENV NEO4J_dbms_directories_logs=/app/neo4j_logs

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy Nginx config and run script
COPY nginx.conf /etc/nginx/nginx.conf
COPY scripts/run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Expose HTTP (Nginx), Neo4j, and Gradio ports
EXPOSE 80 7474 7687 7860

# Set environment variables for Neo4j
# Note: In production, use Docker secrets or environment files for sensitive data
ENV NEO4J_HOME=/neo4j
ENV NEO4J_AUTH=neo4j/password
ENV NEO4J_PLUGINS='["apoc"]'

# Entrypoint: Start all services
ENTRYPOINT ["/app/run.sh"]
