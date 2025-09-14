#!/bin/bash
set -e

# Start Neo4j in the background (if available)
if [ -d "/neo4j" ]; then
    /neo4j/bin/neo4j console &
else
    echo "Neo4j not found, starting without Neo4j..."
fi

# Wait for Neo4j to be ready (if running)
if [ -d "/neo4j" ]; then
    until nc -z localhost 7687; do
      echo "Waiting for Neo4j..."
      sleep 2
    done
fi

# Start Nginx in the foreground
nginx -g 'daemon off;' &


# Start Gradio app in the background
python rag_app.py

wait
