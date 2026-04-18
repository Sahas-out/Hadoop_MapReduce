# Hadoop Co-occurrence Matrix Project Documentation

## 1. Setup and How to Run

### Prerequisites
- Java 8
- Maven (pom.xml is already there in the repo)
- Hadoop 3.4.3 
- make sure JAVA_HOME, HADOOP_HOME env variables are configured  along with PATH having JAVA_HOME/bin and HADOOP_HOME/bin (.env_template is given)

### Build
from root directory
```bash
mvn clean package -DskipTests
```

Output jar:
- `target/hadoop-demo-1.0-SNAPSHOT.jar`

### Input files used
- Corpus: `Wikipedia-EN-20120601_ARTICLES/`
- Stopwords: `stopwords.txt`

---

### Run Task 1 (`Task1TopFrequentWords`)
Generates top-50 frequent words (stopwords removed) and task-1 runtime metrics.

```bash
hadoop jar target/hadoop-demo-1.0-SNAPSHOT.jar \
  com.example.hadoop.Task1TopFrequentWords \
  Wikipedia-EN-20120601_ARTICLES \
  output/task1 \
  stopwords.txt \
  12 # number of splits you want leave blank for default behaviour
```

Main outputs:
- `output/task1/task1_top50_words.txt` # top 50 words
- `output/task1/task1_runtime_metrics.txt` # runtime metrics

---

### Run Task 2 (`Task2PairsCooccurrence`)
Builds co-occurrence matrix using **pairs** approach for `d={1,2,3,4}` and all 3 aggregation modes.

```bash
hadoop jar target/hadoop-demo-1.0-SNAPSHOT.jar \
  com.example.hadoop.Task2PairsCooccurrence \
  Wikipedia-EN-20120601_ARTICLES \
  output/task2 \
  output/task1/task1_top50_words.txt \
  12 \  # number of splits you want leave blank for default behaviour
  1,2,3,4 # distances to evaluvate for during runtime
```

Main outputs:
- Per run directories: `output/task2/task2_pairs_<mode>_d<d>/`
- Comparison report: `output/task2/task2_runtime_comparison.txt`

---

### Run Task 3 (`Task3StripesCooccurrence`)
Builds co-occurrence matrix using **stripes** approach for `d={1,2,3,4}` and all 3 aggregation modes.

```bash
hadoop jar target/hadoop-demo-1.0-SNAPSHOT.jar \
  com.example.hadoop.Task3StripesCooccurrence \
  Wikipedia-EN-20120601_ARTICLES \
  output/task3 \
  output/task1/task1_top50_words.txt \
  12 \  # number of splits you want leave blank for default behaviour
  1,2,3,4 # distances to evaluvate for during runtime
```

Main outputs:
- Per run directories: `output/task3/task3_stripes_<mode>_d<d>/`
- Comparison report: `output/task3/task3_runtime_comparison.txt`

---

## 2. Logic Used in the Three Code Files

## `Task1TopFrequentWords.java`
Goal: extract top-50 frequent words after removing stopwords.

Core logic:
1. Mapper loads `stopwords.txt` from distributed cache.
2. Mapper tokenizes text and emits `(word, 1)` for non-stopwords.
3. IntSumReducer sums local counts.
4. TopKReducer computes global counts and keeps only top 50 using a min-heap.

Mapper pseudocode:
```text
setup:
  load stopwords from cache into set

map(line):
  tokens = tokenize(lowercase(line))
  for token in tokens:
    if token is valid and not stopword:
      emit(token, 1)
```

Reducer pseudocode:
```text
reduce(word, counts):
  total = sum(counts)
  maintain topK heap with (word, total)

cleanup:
  sort heap descending by count
  emit top 50
```

---

## `Task2PairsCooccurrence.java`
Goal: build co-occurrence matrix with **pairs** representation using only top-50 words.

Core logic:
1. Top-50 words file is loaded from distributed cache.
2. For each center word, neighbors within distance `d` are considered.
3. Key format: `"center\tneighbor"`, value: count.
4. Three mapper variants are implemented (difference is **where local aggregation happens**):
   - `NONE`: no local aggregation. Every observed `(center,neighbor)` is emitted immediately as `(pair,1)`. This creates the largest intermediate output.
   - `FUNCTION`: map-function-level aggregation. Counts are aggregated in a temporary hash map for the current `map(document)` call, then emitted once per unique pair in that document.
   - `CLASS`: map-class-level aggregation. Counts are aggregated in a mapper member hash map across all map calls in that mapper task and emitted in `cleanup()`. This usually minimizes shuffle records, at the cost of higher mapper memory usage.
5. Reducer sums counts per pair.

Mapper pseudocode (pairs):
```text
setup:
  load top50 words from cache
  read distance d from config

map(document):
  tokens = tokenize(document)
  for each position i:
    if tokens[i] in top50:
      for j in [i-d, i+d], j != i:
        if tokens[j] in top50:
          pair = (tokens[i], tokens[j])
          count pair (emit directly or local-aggregate depending on mode)
```

Reducer pseudocode:
```text
reduce(pair, counts):
  emit(pair, sum(counts))
```

---

## `Task3StripesCooccurrence.java`
Goal: build co-occurrence matrix with **stripes** representation using only top-50 words.

Core logic:
1. Top-50 words loaded from distributed cache.
2. For each center word, mapper builds a stripe (neighbor -> local count) within distance `d`.
3. Mapper emits `(center, stripeMap)`.
4. Combiner merges stripe maps.
5. Reducer merges all stripes for a center and writes final stripe.
6. Three mapper variants are implemented, and they differ in aggregation scope:
   - `NONE`: emits a stripe per center occurrence immediately. Lowest mapper memory, highest number of emitted stripe records.
   - `FUNCTION`: merges stripes within one `map(document)` call and emits consolidated stripes per center for that document.
   - `CLASS`: merges stripes across the mapper task lifetime and emits only in `cleanup()`. This gives maximum mapper-side compression of intermediate data, with the highest memory pressure.

Mapper pseudocode (stripes):
```text
setup:
  load top50 words from cache
  read distance d from config

map(document):
  tokens = tokenize(document)
  for each center token in top50:
    stripe = {}
    for neighbors within distance d:
      if neighbor in top50:
        stripe[neighbor] += 1
    emit(center, stripe)   # direct or locally aggregated by mode
```

Reducer pseudocode:
```text
reduce(center, stripeMaps):
  merged = {}
  for stripe in stripeMaps:
    merged += stripe (key-wise sum)
  emit(center, serialize(merged))
```

---

## 3. Experimental Setup Design

### A) Split reduction for many small files
- The corpus has many small files (~10,000 files, ~16KB each).
- If each file becomes a separate split, mapper-task overhead becomes very high.
- The jobs compute:
  - `maxSplitSize = totalInputBytes / targetSplits`
  - `minSplitSize = maxSplitSize / 2`
- With `targetSplits=12`, Hadoop creates roughly ~10–12 combined splits, better matching available cores.

Why this improves performance:
- fewer mapper startups and scheduling overhead,
- better task granularity for multi-core execution,
- less framework overhead relative to useful computation.

### B) Whole-document RecordReader (instead of line-based)
- `CombinedWholeFileInputFormat` extends `CombineFileInputFormat<Text,Text>`.
- `WholeFileInCombineRecordReader` returns:
  - key = file path
  - value = full file content
- This preserves co-occurrence windows across line boundaries inside a document while still combining small files into fewer mapper tasks.

### C) Runtime analysis and comparisons

The runtime metrics were read from:
- `output/task1/task1_runtime_metrics.txt`
- `output/task2_1d..4d/task2_runtime_comparison.txt`
- `output/task3_1d..4d/task3_runtime_comparison.txt`

Only runtime metrics are listed below (no split/input-size stats).

#### Task 1 runtime metrics

| Metric | Value (ms) |
| --- | ---: |
| total_job_runtime_ms | 20247 |
| map_setup_ms | 5 |
| map_mapFunction_ms | 5675 |
| map_cleanup_ms | 1 |
| reduce_setup_ms | 0 |
| reduce_reduceFunction_ms | 873 |
| reduce_cleanup_ms | 1 |

#### Task 2 (Pairs) runtimes by aggregation mode and distance

| d | mode | runtime_ms | map_setup_ms | map_mapFunction_ms | map_cleanup_ms | reduce_setup_ms | reduce_reduceFunction_ms | reduce_cleanup_ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | none | 12624 | 6 | 5992 | 1 | 1 | 218 | 1 |
| 1 | function | 10686 | 3 | 5424 | 0 | 0 | 95 | 0 |
| 1 | class | 10715 | 2 | 5518 | 9 | 0 | 50 | 0 |
| 2 | none | 10162 | 5 | 4224 | 1 | 1 | 222 | 0 |
| 2 | function | 8492 | 2 | 4012 | 0 | 0 | 94 | 0 |
| 2 | class | 8484 | 1 | 4169 | 8 | 0 | 53 | 0 |
| 3 | none | 10407 | 4 | 4382 | 1 | 1 | 317 | 0 |
| 3 | function | 8496 | 2 | 4046 | 0 | 0 | 125 | 0 |
| 3 | class | 7477 | 2 | 3996 | 9 | 0 | 58 | 0 |
| 4 | none | 10523 | 5 | 4702 | 1 | 1 | 391 | 0 |
| 4 | function | 8488 | 2 | 4388 | 0 | 0 | 144 | 0 |
| 4 | class | 7473 | 2 | 4060 | 9 | 0 | 58 | 0 |

#### Task 3 (Stripes) runtimes by aggregation mode and distance

| d | mode | runtime_ms | map_setup_ms | map_mapFunction_ms | map_cleanup_ms | reduce_setup_ms | reduce_reduceFunction_ms | reduce_cleanup_ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | none | 11347 | 4 | 5318 | 1 | 0 | 15 | 0 |
| 1 | function | 8495 | 2 | 4361 | 0 | 0 | 13 | 0 |
| 1 | class | 7476 | 1 | 3915 | 7 | 0 | 9 | 0 |
| 2 | none | 11381 | 5 | 5586 | 1 | 0 | 20 | 0 |
| 2 | function | 8552 | 2 | 4181 | 0 | 0 | 16 | 0 |
| 2 | class | 7492 | 2 | 3863 | 53 | 0 | 10 | 0 |
| 3 | none | 12388 | 5 | 6366 | 1 | 0 | 19 | 0 |
| 3 | function | 8497 | 2 | 4450 | 0 | 0 | 18 | 0 |
| 3 | class | 7466 | 1 | 3926 | 7 | 0 | 11 | 0 |
| 4 | none | 13345 | 4 | 6548 | 1 | 0 | 19 | 0 |
| 4 | function | 9497 | 2 | 4709 | 0 | 0 | 17 | 0 |
| 4 | class | 8499 | 2 | 4113 | 8 | 0 | 11 | 0 |

#### Comparison summary: Pairs vs Stripes

- For `d=1`, **Stripes** is faster in all modes.
- For `d=2`, **Pairs** is faster for `none` and `function`, while **Stripes** is faster for `class`.
- For `d=3`, **Pairs** is faster for `none`, near tie for `function`, and **Stripes** is slightly faster for `class`.
- For `d=4`, **Pairs** is faster in all modes.
- Stage behavior: `reduce_reduceFunction_ms` is much larger in **Pairs** (up to 391 ms) than in **Stripes** (9–20 ms), showing that stripes pushes more aggregation before/at shuffle and reduces reducer merge work.

#### Comparison summary: aggregation strategy (`none`, `function`, `class`)

- **Pairs**:
  - `none` is slowest for all distances.
  - `class` is best for `d=2,3,4`; for `d=1`, `function` is marginally better.
  - Local aggregation strongly reduces reducer work (`reduce_reduceFunction_ms` drops from 218–391 to ~50–144).
- **Stripes**:
  - consistent order: `class` fastest, `function` second, `none` slowest for all `d`.
  - `none` gets notably worse as `d` grows (11347 -> 13345 ms), while `class` stays comparatively stable (7476 -> 8499 ms).
