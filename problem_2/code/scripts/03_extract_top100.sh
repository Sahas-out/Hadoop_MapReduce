#!/bin/bash
# 03_extract_top100.sh — Extracts top 100 terms by DF and uploads to HDFS.
#
# Reads:   <project-dir>/output/df_output.tsv   (produced by 02_run_2a.sh)
# Writes:  <project-dir>/output/df_top100.tsv   (local)
#          /user/$USER/friend_df_top100.tsv      (HDFS, used by 04_run_2b.sh)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DF="$SCRIPT_DIR/output/df_output.tsv"
LOCAL_TOP100="$SCRIPT_DIR/output/df_top100.tsv"
HDFS_TOP100=/user/$USER/friend_df_top100.tsv

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/mnt/newstorage/hadoop
export PATH=$JAVA_HOME/bin:$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

if [ ! -f "$LOCAL_DF" ]; then
    echo "ERROR: $LOCAL_DF not found. Run 02_run_2a.sh first."
    exit 1
fi

echo "=== [top100] Extracting top 100 terms ==="

sort -t$'\t' -k2 -rn "$LOCAL_DF" | head -100 > "$LOCAL_TOP100"

echo "Top 100 terms (TERM<tab>DF):"
echo "----------------------------"
cat "$LOCAL_TOP100"

TEMP="/tmp/friend_df_top100_$$.tsv"
cp "$LOCAL_TOP100" "$TEMP"
hdfs dfs -rm -f "$HDFS_TOP100" 2>/dev/null || true
hdfs dfs -put "$TEMP" "$HDFS_TOP100"
rm -f "$TEMP"

echo ""
echo "Local : $LOCAL_TOP100"
echo "HDFS  : $HDFS_TOP100"
echo ""
echo "Next: run  ./scripts/04_run_2b.sh"
