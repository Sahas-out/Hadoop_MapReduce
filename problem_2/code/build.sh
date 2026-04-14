#!/bin/bash
# build.sh — Compiles the Maven project and produces problem2.jar in the project root.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo "=== Building problem2 (2a + 2b) with Maven ==="
cd "$SCRIPT_DIR"
mvn clean package -q

if [ -f "$SCRIPT_DIR/problem2.jar" ]; then
    echo ""
    echo "Build successful → $SCRIPT_DIR/problem2.jar"
    echo "  Entry points:"
    echo "    parta.DocumentFrequency  (Problem 2a — Document Frequency)"
    echo "    partb.TFIDFScorer        (Problem 2b — TF-IDF Scorer)"
else
    echo "ERROR: problem2.jar not found after build."
    exit 1
fi
