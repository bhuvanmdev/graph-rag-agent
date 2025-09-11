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

echo.
echo Starting Neo4j container...
docker run -d ^
    --name neo4j-rag ^
    -p 7474:7474 -p 7687:7687 ^
    -e NEO4J_AUTH=neo4j/password ^
    -e NEO4J_PLUGINS=["apoc","neo4j-vector"] ^
    -v %cd%\neo4j_data:/data ^
    -v %cd%\neo4j_logs:/logs ^
    neo4j:5.15

echo.
echo Waiting for Neo4j to start...
timeout /t 30 /nobreak >nul

echo.
echo Neo4j is starting up...
echo - Web interface: http://localhost:7474
echo - Username: neo4j
echo - Password: password
echo.
echo Checking if Neo4j is ready...
timeout /t 10 /nobreak >nul

echo.
echo Setup complete! Neo4j should be running.
echo You can now run the ingestion pipeline.
echo.
pause
