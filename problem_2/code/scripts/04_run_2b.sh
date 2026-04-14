#!/bin/bash
# 04_run_2b.sh — Runs Problem 2b: TF-IDF Scorer on the full Wikipedia dataset.
#
# Input  (HDFS): /user/$USER/input/wiki_full/Wikipedia-EN-20120601_ARTICLES
#                /user/$USER/friend_df_top100.tsv
# Output (local): <project-dir>/output/tfidf_output.tsv
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JAR="$SCRIPT_DIR/problem2.jar"
LOCAL_OUTPUT="$SCRIPT_DIR/output/tfidf_output.tsv"
HDFS_INPUT=/user/$USER/input/wiki_full/Wikipedia-EN-20120601_ARTICLES
HDFS_OUTPUT=/user/$USER/output/friend_tfidf_output
HDFS_TOP100=/user/$USER/friend_df_top100.tsv

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/mnt/newstorage/hadoop
export PATH=$JAVA_HOME/bin:$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

mkdir -p "$SCRIPT_DIR/output"

if [ ! -f "$JAR" ]; then
    echo "ERROR: $JAR not found. Run ./build.sh first."
    exit 1
fi

if ! hdfs dfs -test -e "$HDFS_TOP100" 2>/dev/null; then
    echo "ERROR: $HDFS_TOP100 not on HDFS. Run 03_extract_top100.sh first."
    exit 1
fi

if ! hdfs dfs -test -e "$HDFS_INPUT" 2>/dev/null; then
    echo "ERROR: $HDFS_INPUT not on HDFS."
    exit 1
fi

hdfs dfs -rm -r "$HDFS_OUTPUT" 2>/dev/null && echo "Removed old HDFS output" || true

echo ""
echo "=== [2b] Running TFIDFScorer (CombineTextInputFormat) ==="
echo "Input        : $HDFS_INPUT"
echo "HDFS Output  : $HDFS_OUTPUT"
echo "DF Top-100   : $HDFS_TOP100"
echo "Local Output : $LOCAL_OUTPUT"
echo ""

START=$(date +%s)

hadoop jar "$JAR" partb.TFIDFScorer \
    "$HDFS_INPUT" \
    "$HDFS_OUTPUT" \
    "$HDFS_TOP100"

END=$(date +%s)

TEMP="/tmp/friend_tfidf_output_$$.tsv"
hdfs dfs -getmerge "$HDFS_OUTPUT" "$TEMP"
mv "$TEMP" "$LOCAL_OUTPUT"

echo ""
echo "=== [2b] Done in $((END-START))s ==="
echo "Output: $LOCAL_OUTPUT"
echo "Total rows: $(wc -l < "$LOCAL_OUTPUT")"
echo ""
echo "Sample rows (ID<tab>TERM<tab>SCORE):"
echo "-------------------------------------"
head -20 "$LOCAL_OUTPUT"
