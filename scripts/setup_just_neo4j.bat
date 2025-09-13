@echo off
echo Starting Neo4j with Docker...
echo.

REM Check if Docker is running
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Pulling Neo4j Docker image...
docker pull neo4j:5.15

echo Creating data directories...
if not exist "neo4j_data" mkdir neo4j_data
if not exist "neo4j_logs" mkdir neo4j_logs

echo Running Neo4j container...
docker run -d ^
    --name neo4j-rag ^
    -p 7474:7474 -p 7687:7687 ^
    -e NEO4J_AUTH=neo4j/password ^
    -e NEO4J_PLUGINS=["apoc","neo4j-vector"] ^
    -v %cd%\neo4j_data:/data ^
    -v %cd%\neo4j_logs:/logs ^
    neo4j:5.15

echo Neo4j is starting up. It may take a few moments for the database to be fully operational.
echo You can access the Neo4j Browser at: http://localhost:7474