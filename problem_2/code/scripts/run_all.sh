#!/bin/bash
# run_all.sh — Builds and runs the full pipeline: build -> 2a -> top100 -> 2b
#
# Prerequisites: Wikipedia data already on HDFS at
#   /user/$USER/input/wiki_full/Wikipedia-EN-20120601_ARTICLES
#   /user/$USER/stopwords.txt
# (These are uploaded by problem2's run_2a_full.sh, or manually via hdfs dfs -put)
#
# Run from the project root:
#   ./scripts/run_all.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo "================================================"
echo " NoSQL Assignment 2 — Friend's Pipeline"
echo " (CombineTextInputFormat — fast run)"
echo "================================================"

echo ""
echo "--- Step 1/4: Build JAR ---"
bash "$PROJECT_DIR/build.sh"

echo ""
echo "--- Step 2/4: Run Problem 2a (Document Frequency) ---"
bash "$SCRIPT_DIR/02_run_2a.sh"

echo ""
echo "--- Step 3/4: Extract top 100 terms ---"
bash "$SCRIPT_DIR/03_extract_top100.sh"

echo ""
echo "--- Step 4/4: Run Problem 2b (TF-IDF Scorer) ---"
bash "$SCRIPT_DIR/04_run_2b.sh"

echo ""
echo "================================================"
echo " All done!"
echo " Results in: $PROJECT_DIR/output/"
echo "   df_output.tsv     — Document Frequency (all terms)"
echo "   df_top100.tsv     — Top 100 terms by DF"
echo "   tfidf_output.tsv  — TF-IDF scores (ID<tab>TERM<tab>SCORE)"
echo "================================================"
