#!/bin/bash
set -e

echo "=== 1. uv run simplecoder --help ==="
uv run simplecoder --help

echo ""
echo "=== 2. uv run simplecoder 'create a hello.py file' ==="
uv run simplecoder "create a hello.py file"

echo ""
echo "=== 3. uv run simplecoder --use-rag 'what does the Agent class do?' ==="
uv run simplecoder --use-rag "what does the Agent class do?"

echo ""
echo "=== 4. uv run simplecoder --use-planning 'create a web server with routes for home and about' ==="
uv run simplecoder --use-planning "create a web server with routes for home and about"

echo ""
echo "=== 5. uv run simplecoder --verbose 'create a hello.py file' ==="
uv run simplecoder --verbose "create a hello.py file"

echo ""
echo "=== 6. uv run simplecoder --interactive (exit to quit) ==="
echo "exit" | uv run simplecoder --interactive

echo ""
echo "=== 7. uv run simplecoder (no task, interactive, exit to quit) ==="
echo "exit" | uv run simplecoder
