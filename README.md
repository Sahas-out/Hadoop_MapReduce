# Word Co-occurrence & TF-IDF at Scale — Hadoop MapReduce

Distributed text-mining pipelines over a **10,000-document Wikipedia corpus**, built from scratch in Java on Hadoop 3.4.3.

The project implements the **Pairs** and **Stripes** co-occurrence algorithms, a **custom `InputFormat`** to solve Hadoop's small-files problem without losing document context, and a **phase-level runtime instrumentation harness** used to benchmark 24 job configurations and quantify exactly where local aggregation pays off.

---

## Highlights

| | |
|---|---|
| **Scale** | ~10,000 files · Wikipedia EN dump (2012-06-01) |
| **Jobs benchmarked** | 24 (2 algorithms × 3 aggregation strategies × 4 window sizes) |
| **Key optimization** | Custom `CombineFileInputFormat` — collapsed ~10,000 map tasks into ~12 |
| **Measured impact** | In-mapper combining cut reducer time by **up to 85%** (391 ms → 58 ms) |
| **Stack** | Java 8 · Hadoop 3.4.3 (HDFS + YARN + MapReduce) · Maven · Apache OpenNLP |

---

## Problem 1 — Co-occurrence Matrix: Pairs vs. Stripes

Given a corpus, compute how often each of the top-50 most frequent words appears within a window of `d` words of every other top-50 word — for `d = {1, 2, 3, 4}`.

### Pipeline

```
Wikipedia corpus ──► Task 1: Top-50 Words ──► top50.txt ──┬──► Task 2: Pairs   ──► matrix + metrics
                     (stopwords via                       │
                      Distributed Cache)                  └──► Task 3: Stripes ──► matrix + metrics
```

Each stage is a standalone MapReduce job; the top-50 vocabulary is passed forward through Hadoop's **Distributed Cache**, so every mapper loads it once in `setup()` rather than re-reading HDFS per record.

### Task 1 — Top-50 frequent words
- Stopword list broadcast via Distributed Cache and filtered **in the mapper**, minimizing intermediate output at the source.
- Combiner reuses the sum reducer (valid — addition is associative and commutative).
- Final reducer runs a **bounded min-heap top-K selection** in `O(n log 50)` instead of sorting the full vocabulary.

### Task 2 — Pairs approach
Emits `(word, neighbor) → 1` for every co-occurrence. Simple and memory-light, but produces one shuffle record per observation.

### Task 3 — Stripes approach
Emits `word → {neighbor: count, ...}` as a `MapWritable`, pre-merging an entire neighborhood into one record. Fewer, larger records; a custom `StripeCombiner` merges partial maps before the shuffle.

### The core experiment — where should local aggregation live?

Each algorithm was implemented **three ways** to isolate the cost of aggregation placement:

| Mode | Aggregation scope | Trade-off |
|---|---|---|
| `NONE` | None — emit every observation | Baseline. Max shuffle volume, max reducer work. |
| `FUNCTION` | A `HashMap` local to a single `map()` call | Collapses duplicates within one document; memory released each call. |
| `CLASS` | A `HashMap` held as a mapper field, flushed in `cleanup()` | **In-mapper combining.** Maximum pre-shuffle compression across the whole map task — at the cost of holding state in heap for the task's lifetime. |

This distinction matters: a Hadoop `Combiner` is a *hint* the framework may skip, whereas in-mapper combining is a **guaranteed** reduction the developer controls — trading a network/reduce cost for a bounded memory risk.

---

## Engineering Deep-Dives

### Solving the small-files problem *without* breaking co-occurrence
The corpus is ~10,000 files of ~16 KB each. With the default `TextInputFormat` that means ~10,000 map tasks, where JVM startup and scheduling overhead dwarf the actual computation.

The standard fix — `CombineTextInputFormat` — bin-packs small files into fewer splits, and Task 1 uses it directly, sizing splits dynamically to the hardware:

```java
maxSplitSize = totalInputBytes / targetSplits;   // targetSplits = 12, matched to available cores
minSplitSize = maxSplitSize / 2;
```

But co-occurrence has a constraint word-counting doesn't: **a line-based reader silently destroys every word pair that spans a line break.** Two words 1 token apart across a newline are no longer neighbors.

So Tasks 2 and 3 use a custom **`CombinedWholeFileInputFormat extends CombineFileInputFormat<Text, Text>`** with a purpose-built `RecordReader` that delivers each file as a *single* record (`key = file path`, `value = full document text`) — while still packing many files into one split. This solves both concerns at once: **task-count efficiency and document-level context integrity.**

### Phase-level runtime instrumentation
Hadoop's built-in counters don't break execution time down by lifecycle phase, so the jobs carry their own. Every `setup()`, `map()`/`reduce()`, and `cleanup()` is wrapped in `System.nanoTime()` and reported through a custom `Counter` enum, then aggregated into a per-task comparison report.

This is what makes the results *diagnostic* rather than anecdotal — it shows not just that a configuration was faster, but **which phase the work moved to.**

---

## Results

Full tables in [`doc.md`](doc.md). Headline findings:

**1. Local aggregation always wins, and `CLASS`-level usually wins outright.**
Ordering was consistent across both algorithms and every window size: `class` ≥ `function` > `none`.

**2. Aggregation doesn't just speed things up — it relocates the work.**
For Pairs, reducer time collapsed as aggregation moved earlier:

| `d` | `none` | `function` | `class` |
|---:|---:|---:|---:|
| 1 | 218 ms | 95 ms | **50 ms** |
| 4 | 391 ms | 144 ms | **58 ms** |

**3. There is no universal winner between Pairs and Stripes — it flips with window size.**
Stripes wins at `d=1`; Pairs wins at `d=4`. The instrumentation explains why: Stripes' reducer time stays flat (~9–20 ms) because map-side merging already did the work, while Pairs' reducer time scales with `d` as raw pair records multiply. Stripes pays that cost up front in map-side memory and `MapWritable` serialization — so which side wins depends on whether the compression saves more than the overhead costs.

**4. Aggregation buys stability, not just speed.**
As `d` grew from 1→4, Stripes with no aggregation degraded 11.3 s → 13.3 s, while `class`-level aggregation held near-flat at 7.5 s → 8.5 s.

---

## Problem 2 — TF-IDF Scoring

A second two-stage pipeline over the same corpus, in [`problem_2/`](problem_2/):

- **Part A — `DocumentFrequency`**: tokenizes, filters stopwords/short tokens/roman numerals, applies **Porter stemming** (Apache OpenNLP), and emits `(term, docId)`. The reducer counts *distinct* document IDs per term to produce DF.
- **Part B — `TFIDFScorer`**: uses the **stripes pattern with in-mapper combining** to build per-document term-frequency maps, loads the DF table from Distributed Cache, and computes a smoothed score:

  ```
  score = tf × log(N / df + 1)
  ```

  The `+1` smoothing keeps the score well-defined at the boundaries where a raw `log(N/df)` would collapse to zero or diverge.

---

## Build & Run

**Prerequisites:** Java 8, Maven, Hadoop 3.4.3, with `JAVA_HOME` and `HADOOP_HOME` exported (see `env_template`).

```bash
mvn clean package -DskipTests
```

**Task 1 — Top-50 words**
```bash
hadoop jar target/hadoop-demo-1.0-SNAPSHOT.jar \
  com.example.hadoop.Task1TopFrequentWords \
  Wikipedia-EN-20120601_ARTICLES output/task1 stopwords.txt 12
```

**Task 2 — Pairs** (runs all 3 aggregation modes × all distances)
```bash
hadoop jar target/hadoop-demo-1.0-SNAPSHOT.jar \
  com.example.hadoop.Task2PairsCooccurrence \
  Wikipedia-EN-20120601_ARTICLES output/task2 \
  output/task1/task1_top50_words.txt 12 1,2,3,4
```

**Task 3 — Stripes**
```bash
hadoop jar target/hadoop-demo-1.0-SNAPSHOT.jar \
  com.example.hadoop.Task3StripesCooccurrence \
  Wikipedia-EN-20120601_ARTICLES output/task3 \
  output/task1/task1_top50_words.txt 12 1,2,3,4
```

Arguments: `<input> <output> <cache-file> [target-splits] [distances-csv]`. Runtime reports are written to `<output>/task*_runtime_comparison.txt`.

---

## Repository Layout

```
src/main/java/com/example/hadoop/
├── Task1TopFrequentWords.java        # Top-K with distributed-cache stopword filtering
├── Task2PairsCooccurrence.java       # Pairs, 3 aggregation modes, benchmark driver
├── Task3StripesCooccurrence.java     # Stripes + custom combiner, 3 aggregation modes
└── CombinedWholeFileInputFormat.java # Custom InputFormat: whole-file records, combined splits

problem_2/code/src/main/java/
├── parta/DocumentFrequency.java      # DF with Porter stemming
└── partb/TFIDFScorer.java            # TF-IDF via stripes + in-mapper combining

doc.md                                # Full methodology and benchmark tables
output/                               # Generated matrices and runtime reports
```

---

## Concepts Demonstrated

Pairs & Stripes design patterns · in-mapper combining vs. framework Combiners · custom `InputFormat` and `RecordReader` · Distributed Cache side-data broadcast · the small-files problem and split sizing · bounded top-K selection · custom Hadoop Counters for performance analysis · multi-job pipeline orchestration · empirical benchmarking and trade-off analysis.
