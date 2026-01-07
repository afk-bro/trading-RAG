#!/bin/bash

# Trading RAG Pipeline - Environment Setup Script
# This script sets up and starts the development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Trading RAG Pipeline - Setup Script  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}$name is ready!${NC}"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e "${RED}$name failed to start after $max_attempts attempts${NC}"
    return 1
}

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker and Docker Compose are installed.${NC}"

# Determine docker compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Step 2: Check for .env file
echo -e "${BLUE}Step 2: Checking environment configuration...${NC}"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        echo -e "${YELLOW}Creating .env from .env.example...${NC}"
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        echo -e "${YELLOW}Please edit .env with your actual configuration values.${NC}"
    else
        echo -e "${YELLOW}No .env file found. Creating template...${NC}"
        cat > "$PROJECT_DIR/.env" << 'EOF'
# Supabase Configuration (REQUIRED)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Database Connection (REQUIRED)
# Get from Supabase Dashboard: Project Settings > Database > Connection string
# Use transaction pooler (port 6543) for most deployments
# DATABASE_URL=postgresql://postgres.your-project-ref:your-password@aws-0-us-east-1.pooler.supabase.com:6543/postgres

# OpenRouter Configuration (OPTIONAL - only needed for mode=answer queries)
# OPENROUTER_API_KEY=your-openrouter-api-key

# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_ACTIVE=kb_nomic_embed_text_v1

# Ollama Configuration
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
EMBED_MODEL=nomic-embed-text

# Service Configuration
SERVICE_PORT=8000
SERVICE_HOST=0.0.0.0
LOG_LEVEL=INFO

# Optional: YouTube API (for metadata enrichment)
# YOUTUBE_API_KEY=your-youtube-api-key
EOF
        echo -e "${YELLOW}Please edit .env with your actual configuration values.${NC}"
    fi
fi

# Step 3: Create Docker network if it doesn't exist
echo -e "${BLUE}Step 3: Setting up Docker network...${NC}"

# Clean up any prefixed network from previous docker-compose runs
docker network rm trading-rag_rag-net 2>/dev/null || true

# Create the external network if it doesn't exist
if ! docker network ls --format '{{.Name}}' | grep -q "^rag-net$"; then
    docker network create rag-net
    echo -e "${GREEN}Created rag-net Docker network.${NC}"
else
    echo -e "${GREEN}rag-net Docker network already exists.${NC}"
fi

# Step 4: Start infrastructure services (Qdrant, Ollama)
echo -e "${BLUE}Step 4: Starting infrastructure services...${NC}"

cd "$PROJECT_DIR"
$DOCKER_COMPOSE -f docker-compose.rag.yml up -d qdrant ollama

# Step 5: Wait for services to be ready
echo -e "${BLUE}Step 5: Waiting for services to be ready...${NC}"

wait_for_service "Qdrant" "http://localhost:6333/health" 30
wait_for_service "Ollama" "http://localhost:11434/api/tags" 60

# Step 6: Pull the embedding model
echo -e "${BLUE}Step 6: Pulling embedding model (nomic-embed-text)...${NC}"

# Check if model is already pulled
if docker exec trading-rag-ollama ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    echo -e "${GREEN}nomic-embed-text model already available.${NC}"
else
    echo -e "${YELLOW}Pulling nomic-embed-text model (this may take a few minutes)...${NC}"
    docker exec trading-rag-ollama ollama pull nomic-embed-text
    echo -e "${GREEN}nomic-embed-text model pulled successfully.${NC}"
fi

# Step 7: Create Python virtual environment (if running service locally)
echo -e "${BLUE}Step 7: Setting up Python environment...${NC}"

if [ -d "$PROJECT_DIR/app" ]; then
    cd "$PROJECT_DIR"

    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Creating Python virtual environment...${NC}"
        python3 -m venv .venv
    fi

    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    source .venv/bin/activate

    if [ -f "requirements.txt" ]; then
        pip install --quiet --upgrade pip
        pip install --quiet -r requirements.txt
        echo -e "${GREEN}Python dependencies installed.${NC}"
    else
        echo -e "${YELLOW}No requirements.txt found. Skipping dependency installation.${NC}"
    fi

    deactivate
fi

# Step 8: Start the FastAPI service
echo -e "${BLUE}Step 8: Starting FastAPI service...${NC}"

$DOCKER_COMPOSE -f docker-compose.rag.yml up -d --build trading-rag-svc

# Give service a moment to start and produce logs
sleep 5

# Show diagnostic info before waiting
echo -e "${YELLOW}=== DIAGNOSTIC INFO ===${NC}"
echo "Container status:"
docker ps -a --filter "name=trading-rag" --format "table {{.Names}}\t{{.Status}}"
echo ""
echo "Container logs (last 100 lines):"
docker logs trading-rag-svc --tail 100 2>&1
echo ""
echo "=== END DIAGNOSTIC INFO ==="
echo ""

wait_for_service "Trading RAG Service" "http://localhost:8000/health" 60

# Print summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Environment Setup Complete!          ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Services running:${NC}"
echo -e "  - Qdrant:              http://localhost:6333"
echo -e "  - Qdrant Dashboard:    http://localhost:6333/dashboard"
echo -e "  - Ollama:              http://localhost:11434"
echo -e "  - Trading RAG Service: http://localhost:8000"
echo -e "  - API Documentation:   http://localhost:8000/docs"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  - View logs:           $DOCKER_COMPOSE -f docker-compose.rag.yml logs -f"
echo -e "  - Stop services:       $DOCKER_COMPOSE -f docker-compose.rag.yml down"
echo -e "  - Restart service:     $DOCKER_COMPOSE -f docker-compose.rag.yml restart trading-rag-svc"
echo -e "  - Check health:        curl http://localhost:8000/health"
echo ""
echo -e "${YELLOW}Remember to:${NC}"
echo -e "  1. Update .env with your Supabase credentials and DATABASE_URL"
echo -e "     - Get DATABASE_URL from Supabase Dashboard > Project Settings > Database"
echo -e "     - Use transaction pooler URL (port 6543) for most deployments"
echo -e "  2. Create a workspace in Supabase (required before ingesting documents)"
echo -e "  3. (Optional) Add OPENROUTER_API_KEY for LLM answer generation"
echo -e "  4. Configure n8n workflow for YouTube ingestion (optional)"
echo ""
