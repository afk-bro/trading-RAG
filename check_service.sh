#!/bin/bash
# Check service status

echo "=== Checking ports ==="
ss -tlnp 2>/dev/null | grep -E "(8000|6333|11434)" || netstat -tlnp 2>/dev/null | grep -E "(8000|6333|11434)" || echo "Cannot check ports"

echo ""
echo "=== Checking container status ==="
docker ps -a --filter "name=trading-rag" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Trading RAG SVC logs (last 50 lines) ==="
docker logs trading-rag-svc --tail 50 2>&1

echo ""
echo "=== Test connections ==="
curl -s -o /dev/null -w "Qdrant: %{http_code}\n" http://localhost:6333/health 2>/dev/null || echo "Qdrant: unreachable"
curl -s -o /dev/null -w "Ollama: %{http_code}\n" http://localhost:11434/api/tags 2>/dev/null || echo "Ollama: unreachable"
curl -s -o /dev/null -w "Trading RAG root: %{http_code}\n" http://localhost:8000/ 2>/dev/null || echo "Trading RAG: unreachable"
curl -s -o /dev/null -w "Trading RAG health: %{http_code}\n" http://localhost:8000/health 2>/dev/null || echo "Trading RAG health: unreachable"
