#!/bin/bash

ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!
sleep 5
if ! ollama list | grep -q 'llama3.2'; then
    echo "Model 'llama3.2' not found. Pulling it now..."
    ollama pull llama3.2
    echo "Pulled"
else
    echo "Model 'llama3.2' is already available."
fi
wait $OLLAMA_PID