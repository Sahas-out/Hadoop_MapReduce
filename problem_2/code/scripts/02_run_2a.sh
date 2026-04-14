#!/bin/bash
# 02_run_2a.sh — Runs Problem 2a: Document Frequency on the full Wikipedia dataset.
#
# Input  (HDFS): /user/$USER/input/wiki_full/Wikipedia-EN-20120601_ARTICLES
# Output (local): <project-dir>/output/df_output.tsv
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JAR="$SCRIPT_DIR/problem2.jar"
LOCAL_OUTPUT="$SCRIPT_DIR/output/df_output.tsv"
HDFS_INPUT=/user/$USER/input/wiki_full/Wikipedia-EN-20120601_ARTICLES
HDFS_OUTPUT=/user/$USER/output/friend_df_output
HDFS_STOPWORDS=/user/$USER/stopwords.txt

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/mnt/newstorage/hadoop
export PATH=$JAVA_HOME/bin:$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

mkdir -p "$SCRIPT_DIR/output"

if [ ! -f "$JAR" ]; then
    echo "ERROR: $JAR not found. Run ./build.sh first."
    exit 1
fi

if ! hdfs dfs -test -e "$HDFS_STOPWORDS" 2>/dev/null; then
    echo "ERROR: $HDFS_STOPWORDS not on HDFS."
    echo "The Wikipedia data should already be on HDFS from problem2 runs."
    echo "If not, run: hdfs dfs -put <path-to-stopwords.txt> $HDFS_STOPWORDS"
    exit 1
fi

if ! hdfs dfs -test -e "$HDFS_INPUT" 2>/dev/null; then
    echo "ERROR: $HDFS_INPUT not on HDFS."
    echo "The Wikipedia data should already be on HDFS from problem2 runs."
    exit 1
fi

hdfs dfs -rm -r "$HDFS_OUTPUT" 2>/dev/null && echo "Removed old HDFS output" || true

echo ""
echo "=== [2a] Running DocumentFrequency (CombineTextInputFormat) ==="
echo "Input       : $HDFS_INPUT"
echo "HDFS Output : $HDFS_OUTPUT"
echo "Local Output: $LOCAL_OUTPUT"
echo ""

START=$(date +%s)

hadoop jar "$JAR" parta.DocumentFrequency \
    "$HDFS_INPUT" \
    "$HDFS_OUTPUT" \
    "$HDFS_STOPWORDS"

END=$(date +%s)

TEMP="/tmp/friend_df_output_$$.tsv"
hdfs dfs -getmerge "$HDFS_OUTPUT" "$TEMP"
mv "$TEMP" "$LOCAL_OUTPUT"

echo ""
echo "=== [2a] Done in $((END-START))s ==="
echo "Output: $LOCAL_OUTPUT"
echo "Total terms: $(wc -l < "$LOCAL_OUTPUT")"
echo ""
echo "Top 20 terms by Document Frequency:"
sort -t$'\t' -k2 -rn "$LOCAL_OUTPUT" | head -20
echo ""
echo "Next: run  ./scripts/03_extract_top100.sh"
