#!/bin/bash
set -e

# Use the exact secret names. The script will fail if NEO4J_PASSWORD is not set.
NEO4J_USER="${NEO4J_USERNAME:-neo4j}"
NEO4J_PASS="${NEO4J_PASSWORD}"

if [ -z "$NEO4J_PASS" ]; then
  echo "CRITICAL ERROR: The secret NEO4J_PASSWORD must be set in your Hugging Face Space."
  exit 1
fi

# === Neo4j Security and Initialization ===
# 1. Set the initial password for the 'neo4j' database user.
echo "Setting initial Neo4j password..."
/neo4j/bin/neo4j-admin dbms set-initial-password "$NEO4J_PASS"

# === Start Services ===
# 2. Start Neo4j in the background
echo "Starting Neo4j database..."
/neo4j/bin/neo4j console &

# 3. Wait for Neo4j to be ready
# The Gradio app will fail if it starts before the database is available.
# We will ping the Bolt port (7687) until it's open.
echo "Waiting for Neo4j to start..."
while ! nc -z localhost 7687; do
  sleep 0.5 # wait for 500 milliseconds before check again
  echo -n "."
done
echo "Neo4j is ready!"

# 4. Start the main Gradio application in the foreground
# This command keeps the container running.
echo "Starting Gradio application (rag_app.py)..."
python rag_app.py
