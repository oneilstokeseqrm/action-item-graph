#!/usr/bin/env bash
#
# Package the action-item-graph Lambda forwarder into a deployment zip.
#
# Usage: ./scripts/package_lambda.sh
# Output: dist/action-item-graph-ingest.zip
#
# Dependencies are installed for aarch64 (arm64) Linux target.
# Only the lambda_ingest/ subpackage is included — no openai, neo4j, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR=$(mktemp -d)
OUTPUT_DIR="$PROJECT_DIR/dist"
OUTPUT="$OUTPUT_DIR/action-item-graph-ingest.zip"

trap 'rm -rf "$BUILD_DIR"' EXIT

echo "=== Building Lambda package ==="
echo "Build dir: $BUILD_DIR"

# 1. Install minimal dependencies for arm64 Lambda runtime
echo "--- Installing dependencies ---"
uv pip install \
    --target "$BUILD_DIR" \
    --python-platform aarch64-manylinux2014 \
    --python-version 3.11 \
    --only-binary :all: \
    pydantic pydantic-settings httpx "aws-lambda-powertools[tracer]"

# 2. Copy only the lambda_ingest subpackage
echo "--- Copying Lambda code ---"
mkdir -p "$BUILD_DIR/action_item_graph/lambda_ingest"
cp "$PROJECT_DIR/src/action_item_graph/__init__.py" "$BUILD_DIR/action_item_graph/" 2>/dev/null || touch "$BUILD_DIR/action_item_graph/__init__.py"
cp "$PROJECT_DIR/src/action_item_graph/lambda_ingest/"*.py "$BUILD_DIR/action_item_graph/lambda_ingest/"

# 3. Create zip
echo "--- Creating zip ---"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT"
cd "$BUILD_DIR"
zip -r9 "$OUTPUT" . -x "*.pyc" "__pycache__/*"

echo "=== Done ==="
echo "Output: $OUTPUT"
du -h "$OUTPUT"
