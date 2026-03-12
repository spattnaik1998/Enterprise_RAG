#!/bin/bash
cd "C:\Users\91838\Downloads\Enterprise_RAG"

# Start server
echo "[Server] Starting FastAPI server..."
uvicorn app.server:app --reload --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "[Server] Waiting for server startup..."
sleep 10

# Test /agent endpoint
echo ""
echo "[Test] Testing /agent endpoint..."
AGENT_RESPONSE=$(curl -s http://localhost:8000/agent)
if echo "$AGENT_RESPONSE" | grep -q "<!DOCTYPE\|<html"; then
    echo "[Test] ✓ /agent endpoint returns HTML"
    AGENT_SIZE=${#AGENT_RESPONSE}
    echo "[Test]   Response size: $AGENT_SIZE bytes"
else
    echo "[Test] ✗ /agent endpoint error:"
    echo "$AGENT_RESPONSE"
fi

# Test /rag endpoint
echo ""
echo "[Test] Testing /rag endpoint..."
RAG_RESPONSE=$(curl -s http://localhost:8000/rag)
if echo "$RAG_RESPONSE" | grep -q "<!DOCTYPE\|<html"; then
    echo "[Test] ✓ /rag endpoint returns HTML"
else
    echo "[Test] ✗ /rag endpoint error"
fi

# Kill server
echo ""
echo "[Server] Cleaning up..."
kill $SERVER_PID 2>/dev/null || true

echo "[Test] Done!"
