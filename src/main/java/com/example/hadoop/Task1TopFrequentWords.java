package com.example.hadoop;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Task1TopFrequentWords {

    private static final int TOP_K = 50;
    private static final int DEFAULT_TARGET_SPLITS = 12;

    private enum RuntimeCounters {
        MAP_SETUP_NANOS,
        MAP_MAPFUNC_NANOS,
        MAP_CLEANUP_NANOS,
        REDUCE_SETUP_NANOS,
        REDUCE_REDUCEFUNC_NANOS,
        REDUCE_CLEANUP_NANOS
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final Text wordOut = new Text();
        private final Set<String> stopwords = new HashSet<String>();

        @Override
        protected void setup(Context context) throws IOException {
            long startNanos = System.nanoTime();
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) {
                throw new IOException("Missing distributed cache file for stopwords.");
            }

            for (URI cacheFile : cacheFiles) {
                loadStopwordsFromCache(cacheFile);
            }
            context.getCounter(RuntimeCounters.MAP_SETUP_NANOS).increment(System.nanoTime() - startNanos);
        }

        private void loadStopwordsFromCache(URI cacheFile) throws IOException {
            Path cachePath = new Path(cacheFile.getPath());
            File localFile = new File(cachePath.getName());

            BufferedReader reader;
            if (localFile.exists()) {
                reader = new BufferedReader(
                        new InputStreamReader(new java.io.FileInputStream(localFile), StandardCharsets.UTF_8));
            } else if (new File(cacheFile.getPath()).exists()) {
                reader = new BufferedReader(
                        new InputStreamReader(new java.io.FileInputStream(new File(cacheFile.getPath())), StandardCharsets.UTF_8));
            } else {
                throw new FileNotFoundException("Could not find cached stopwords file: " + cacheFile);
            }

            try {
                String line;
                while ((line = reader.readLine()) != null) {
                    String stopword = line.trim().toLowerCase();
                    if (!stopword.isEmpty()) {
                        stopwords.add(stopword);
                    }
                }
            } finally {
                reader.close();
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");
            for (String token : tokens) {
                if (!token.isEmpty() && !stopwords.contains(token)) {
                    wordOut.set(token);
                    context.write(wordOut, ONE);
                }
            }
            context.getCounter(RuntimeCounters.MAP_MAPFUNC_NANOS).increment(System.nanoTime() - startNanos);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            context.getCounter(RuntimeCounters.MAP_CLEANUP_NANOS).increment(System.nanoTime() - startNanos);
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable sumOut = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            sumOut.set(sum);
            context.write(key, sumOut);
        }
    }

    public static class TopKReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final Comparator<WordCount> minHeapComparator = new Comparator<WordCount>() {
            @Override
            public int compare(WordCount left, WordCount right) {
                int countCompare = Integer.compare(left.count, right.count);
                if (countCompare != 0) {
                    return countCompare;
                }
                return right.word.compareTo(left.word);
            }
        };

        private final PriorityQueue<WordCount> topWords = new PriorityQueue<WordCount>(TOP_K, minHeapComparator);

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            context.getCounter(RuntimeCounters.REDUCE_SETUP_NANOS).increment(System.nanoTime() - startNanos);
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }

            WordCount candidate = new WordCount(key.toString(), sum);
            if (topWords.size() < TOP_K) {
                topWords.offer(candidate);
            } else if (minHeapComparator.compare(candidate, topWords.peek()) > 0) {
                topWords.poll();
                topWords.offer(candidate);
            }
            context.getCounter(RuntimeCounters.REDUCE_REDUCEFUNC_NANOS).increment(System.nanoTime() - startNanos);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            List<WordCount> sortedTopWords = new ArrayList<WordCount>(topWords);
            sortedTopWords.sort(new Comparator<WordCount>() {
                @Override
                public int compare(WordCount left, WordCount right) {
                    int countCompare = Integer.compare(right.count, left.count);
                    if (countCompare != 0) {
                        return countCompare;
                    }
                    return left.word.compareTo(right.word);
                }
            });

            IntWritable countOut = new IntWritable();
            Text wordOut = new Text();
            for (WordCount entry : sortedTopWords) {
                wordOut.set(entry.word);
                countOut.set(entry.count);
                context.write(wordOut, countOut);
            }
            context.getCounter(RuntimeCounters.REDUCE_CLEANUP_NANOS).increment(System.nanoTime() - startNanos);
        }
    }

    private static final class WordCount {
        private final String word;
        private final int count;

        private WordCount(String word, int count) {
            this.word = word;
            this.count = count;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3 || args.length > 4) {
            System.err.println("Usage: Task1TopFrequentWords <input-path> <output-path> <stopwords-path> [target-splits]");
            System.exit(1);
        }

        int targetSplits = args.length == 4 ? Integer.parseInt(args[3]) : DEFAULT_TARGET_SPLITS;
        if (targetSplits <= 0) {
            throw new IllegalArgumentException("target-splits must be a positive integer.");
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "task1-top50-frequent-words");

        job.setJarByClass(Task1TopFrequentWords.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(TopKReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        Path inputPath = new Path(args[0]);
        Path outputPath = new Path(args[1]);
        Path stopwordsPath = new Path(args[2]);

        job.setInputFormatClass(CombineTextInputFormat.class);
        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);
        job.addCacheFile(stopwordsPath.toUri());

        FileSystem fs = inputPath.getFileSystem(conf);
        ContentSummary contentSummary = fs.getContentSummary(inputPath);
        long totalInputBytes = contentSummary.getLength();
        long maxSplitSize = Math.max(1L, (long) Math.ceil((double) totalInputBytes / targetSplits));
        long minSplitSize = Math.max(1L, maxSplitSize / 2);
        CombineTextInputFormat.setMaxInputSplitSize(job, maxSplitSize);
        CombineTextInputFormat.setMinInputSplitSize(job, minSplitSize);

        job.setNumReduceTasks(1);

        long startNanos = System.nanoTime();
        boolean success = job.waitForCompletion(true);
        long endNanos = System.nanoTime();
        long totalRuntimeMs = (endNanos - startNanos) / 1_000_000L;
        if (success) {
            writeTop50File(fs, outputPath);
            writeRuntimeMetricsFile(job, fs, outputPath, totalRuntimeMs, targetSplits, totalInputBytes, minSplitSize, maxSplitSize);
        }
        System.out.printf("Task1 runtime_ms=%d%n", totalRuntimeMs);

        System.exit(success ? 0 : 1);
    }

    private static void writeTop50File(FileSystem fs, Path outputPath) throws IOException {
        Path top50Path = new Path(outputPath, "task1_top50_words.txt");
        FSDataOutputStream outputStream = fs.create(top50Path, true);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream, StandardCharsets.UTF_8));
        try {
            FileStatus[] topWordFiles = fs.globStatus(new Path(outputPath, "part-r-*"));
            if (topWordFiles != null) {
                for (FileStatus status : topWordFiles) {
                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(fs.open(status.getPath()), StandardCharsets.UTF_8));
                    try {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            writer.write(line);
                            writer.newLine();
                        }
                    } finally {
                        reader.close();
                    }
                }
            }
        } finally {
            writer.close();
        }
    }

    private static void writeRuntimeMetricsFile(
            Job job,
            FileSystem fs,
            Path outputPath,
            long totalRuntimeMs,
            int targetSplits,
            long totalInputBytes,
            long minSplitSize,
            long maxSplitSize) throws IOException {

        Path summaryPath = new Path(outputPath, "task1_runtime_metrics.txt");
        FSDataOutputStream outputStream = fs.create(summaryPath, true);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream, StandardCharsets.UTF_8));
        try {
            writer.write("runtime_metrics_ms");
            writer.newLine();
            writer.write("total_job_runtime_ms\t" + totalRuntimeMs);
            writer.newLine();
            writer.write("map_setup_ms\t" + toMillis(job, RuntimeCounters.MAP_SETUP_NANOS));
            writer.newLine();
            writer.write("map_mapFunction_ms\t" + toMillis(job, RuntimeCounters.MAP_MAPFUNC_NANOS));
            writer.newLine();
            writer.write("map_cleanup_ms\t" + toMillis(job, RuntimeCounters.MAP_CLEANUP_NANOS));
            writer.newLine();
            writer.write("reduce_setup_ms\t" + toMillis(job, RuntimeCounters.REDUCE_SETUP_NANOS));
            writer.newLine();
            writer.write("reduce_reduceFunction_ms\t" + toMillis(job, RuntimeCounters.REDUCE_REDUCEFUNC_NANOS));
            writer.newLine();
            writer.write("reduce_cleanup_ms\t" + toMillis(job, RuntimeCounters.REDUCE_CLEANUP_NANOS));
            writer.newLine();
            writer.write("target_splits\t" + targetSplits);
            writer.newLine();
            writer.write("total_input_bytes\t" + totalInputBytes);
            writer.newLine();
            writer.write("min_split_size_bytes\t" + minSplitSize);
            writer.newLine();
            writer.write("max_split_size_bytes\t" + maxSplitSize);
            writer.newLine();
        } finally {
            writer.close();
        }
    }

    private static long toMillis(Job job, RuntimeCounters counter) throws IOException {
        Counter value = job.getCounters().findCounter(counter);
        return value.getValue() / 1_000_000L;
    }
}
