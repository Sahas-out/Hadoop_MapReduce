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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Task2PairsCooccurrence {

    private static final int DEFAULT_TARGET_SPLITS = 12;
    private static final int[] DEFAULT_DISTANCES = new int[] {1, 2, 3, 4};

    private enum AggregationMode {
        NONE,
        FUNCTION,
        CLASS
    }

    private enum RuntimeCounters {
        MAP_SETUP_NANOS,
        MAP_MAPFUNC_NANOS,
        MAP_CLEANUP_NANOS,
        REDUCE_SETUP_NANOS,
        REDUCE_REDUCEFUNC_NANOS,
        REDUCE_CLEANUP_NANOS
    }

    private abstract static class BasePairsMapper extends Mapper<Object, Text, Text, IntWritable> {
        protected static final IntWritable ONE = new IntWritable(1);
        protected final Set<String> frequentWords = new HashSet<String>();
        protected int distance;

        @Override
        protected void setup(Context context) throws IOException {
            long startNanos = System.nanoTime();
            distance = context.getConfiguration().getInt("cooccur.distance", 1);
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) {
                throw new IOException("Missing distributed cache file for top-50 words.");
            }
            for (URI cacheFile : cacheFiles) {
                loadFrequentWords(cacheFile);
            }
            context.getCounter(RuntimeCounters.MAP_SETUP_NANOS).increment(System.nanoTime() - startNanos);
        }

        private void loadFrequentWords(URI cacheFile) throws IOException {
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
                throw new FileNotFoundException("Could not find cached top-50 file: " + cacheFile);
            }

            try {
                String line;
                while ((line = reader.readLine()) != null) {
                    String trimmed = line.trim().toLowerCase();
                    if (trimmed.isEmpty()) {
                        continue;
                    }
                    String[] parts = trimmed.split("\\s+");
                    if (parts.length > 0) {
                        frequentWords.add(parts[0]);
                    }
                }
            } finally {
                reader.close();
            }
        }

        protected String[] tokenize(String text) {
            return text.toLowerCase().split("[^a-zA-Z]+");
        }

        protected boolean validWord(String token) {
            return !token.isEmpty() && frequentWords.contains(token);
        }

        protected void generatePairCounts(String[] tokens, Map<String, Integer> counts) {
            for (int i = 0; i < tokens.length; i++) {
                String center = tokens[i];
                if (!validWord(center)) {
                    continue;
                }
                int start = Math.max(0, i - distance);
                int end = Math.min(tokens.length - 1, i + distance);
                for (int j = start; j <= end; j++) {
                    if (i == j) {
                        continue;
                    }
                    String neighbor = tokens[j];
                    if (!validWord(neighbor)) {
                        continue;
                    }
                    String pairKey = center + "\t" + neighbor;
                    Integer current = counts.get(pairKey);
                    counts.put(pairKey, current == null ? 1 : current + 1);
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            onCleanup(context);
            context.getCounter(RuntimeCounters.MAP_CLEANUP_NANOS).increment(System.nanoTime() - startNanos);
        }

        protected void onCleanup(Context context) throws IOException, InterruptedException {
            // default no-op
        }
    }

    public static class PairsMapperNoLocalAggregation extends BasePairsMapper {
        private final Text pairOut = new Text();

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            String[] tokens = tokenize(value.toString());
            for (int i = 0; i < tokens.length; i++) {
                String center = tokens[i];
                if (!validWord(center)) {
                    continue;
                }
                int start = Math.max(0, i - distance);
                int end = Math.min(tokens.length - 1, i + distance);
                for (int j = start; j <= end; j++) {
                    if (i == j) {
                        continue;
                    }
                    String neighbor = tokens[j];
                    if (!validWord(neighbor)) {
                        continue;
                    }
                    pairOut.set(center + "\t" + neighbor);
                    context.write(pairOut, ONE);
                }
            }
            context.getCounter(RuntimeCounters.MAP_MAPFUNC_NANOS).increment(System.nanoTime() - startNanos);
        }
    }

    public static class PairsMapperFunctionAggregation extends BasePairsMapper {
        private final Text pairOut = new Text();
        private final IntWritable countOut = new IntWritable();

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            String[] tokens = tokenize(value.toString());
            Map<String, Integer> localCounts = new HashMap<String, Integer>();
            generatePairCounts(tokens, localCounts);
            for (Map.Entry<String, Integer> entry : localCounts.entrySet()) {
                pairOut.set(entry.getKey());
                countOut.set(entry.getValue());
                context.write(pairOut, countOut);
            }
            context.getCounter(RuntimeCounters.MAP_MAPFUNC_NANOS).increment(System.nanoTime() - startNanos);
        }
    }

    public static class PairsMapperClassAggregation extends BasePairsMapper {
        private final Map<String, Integer> classCounts = new HashMap<String, Integer>();
        private final Text pairOut = new Text();
        private final IntWritable countOut = new IntWritable();

        @Override
        public void map(Object key, Text value, Context context) {
            long startNanos = System.nanoTime();
            String[] tokens = tokenize(value.toString());
            generatePairCounts(tokens, classCounts);
            context.getCounter(RuntimeCounters.MAP_MAPFUNC_NANOS).increment(System.nanoTime() - startNanos);
        }

        @Override
        protected void onCleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
                pairOut.set(entry.getKey());
                countOut.set(entry.getValue());
                context.write(pairOut, countOut);
            }
            classCounts.clear();
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable sumOut = new IntWritable();

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
            sumOut.set(sum);
            context.write(key, sumOut);
            context.getCounter(RuntimeCounters.REDUCE_REDUCEFUNC_NANOS).increment(System.nanoTime() - startNanos);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            long startNanos = System.nanoTime();
            context.getCounter(RuntimeCounters.REDUCE_CLEANUP_NANOS).increment(System.nanoTime() - startNanos);
        }
    }

    private static final class RunStats {
        private final AggregationMode mode;
        private final int distance;
        private final long runtimeMs;
        private final long mapSetupMs;
        private final long mapMapMs;
        private final long mapCleanupMs;
        private final long reduceSetupMs;
        private final long reduceReduceMs;
        private final long reduceCleanupMs;
        private final Path outputPath;

        private RunStats(
                AggregationMode mode,
                int distance,
                long runtimeMs,
                long mapSetupMs,
                long mapMapMs,
                long mapCleanupMs,
                long reduceSetupMs,
                long reduceReduceMs,
                long reduceCleanupMs,
                Path outputPath) {
            this.mode = mode;
            this.distance = distance;
            this.runtimeMs = runtimeMs;
            this.mapSetupMs = mapSetupMs;
            this.mapMapMs = mapMapMs;
            this.mapCleanupMs = mapCleanupMs;
            this.reduceSetupMs = reduceSetupMs;
            this.reduceReduceMs = reduceReduceMs;
            this.reduceCleanupMs = reduceCleanupMs;
            this.outputPath = outputPath;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3 || args.length > 5) {
            System.err.println(
                    "Usage: Task2PairsCooccurrence <input-path> <output-base-path> <top50-path> [target-splits] [distances-csv]");
            System.exit(1);
        }

        Path inputPath = new Path(args[0]);
        Path outputBasePath = new Path(args[1]);
        Path top50Path = new Path(args[2]);
        int targetSplits = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_TARGET_SPLITS;
        int[] distances = args.length == 5 ? parseDistances(args[4]) : DEFAULT_DISTANCES;

        if (targetSplits <= 0) {
            throw new IllegalArgumentException("target-splits must be a positive integer.");
        }

        Configuration baseConf = new Configuration();
        FileSystem fs = inputPath.getFileSystem(baseConf);
        fs.mkdirs(outputBasePath);

        ContentSummary contentSummary = fs.getContentSummary(inputPath);
        long totalInputBytes = contentSummary.getLength();
        long maxSplitSize = Math.max(1L, (long) Math.ceil((double) totalInputBytes / targetSplits));
        long minSplitSize = Math.max(1L, maxSplitSize / 2);

        List<RunStats> allRuns = new ArrayList<RunStats>();
        AggregationMode[] modes = new AggregationMode[] {AggregationMode.NONE, AggregationMode.FUNCTION, AggregationMode.CLASS};
        for (AggregationMode mode : modes) {
            for (int distance : distances) {
                RunStats runStats = runJob(
                        inputPath,
                        outputBasePath,
                        top50Path,
                        mode,
                        distance,
                        targetSplits,
                        minSplitSize,
                        maxSplitSize,
                        baseConf,
                        fs);
                allRuns.add(runStats);
                System.out.printf(
                        "Task2 pairs mode=%s d=%d runtime_ms=%d output=%s%n",
                        mode.name().toLowerCase(),
                        distance,
                        runStats.runtimeMs,
                        runStats.outputPath.toString());
            }
        }

        writeRuntimeReport(
                fs,
                outputBasePath,
                allRuns,
                targetSplits,
                totalInputBytes,
                minSplitSize,
                maxSplitSize,
                distances);
    }

    private static RunStats runJob(
            Path inputPath,
            Path outputBasePath,
            Path top50Path,
            AggregationMode mode,
            int distance,
            int targetSplits,
            long minSplitSize,
            long maxSplitSize,
            Configuration baseConf,
            FileSystem fs) throws Exception {

        Configuration conf = new Configuration(baseConf);
        conf.setInt("cooccur.distance", distance);
        conf.set("cooccur.aggregation.mode", mode.name().toLowerCase());
        conf.setInt("cooccur.target.splits", targetSplits);

        String outputDirName = String.format("task2_pairs_%s_d%d", mode.name().toLowerCase(), distance);
        Path outputPath = new Path(outputBasePath, outputDirName);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }

        Job job = Job.getInstance(conf, "task2-pairs-" + mode.name().toLowerCase() + "-d" + distance);
        job.setJarByClass(Task2PairsCooccurrence.class);

        if (mode == AggregationMode.NONE) {
            job.setMapperClass(PairsMapperNoLocalAggregation.class);
        } else if (mode == AggregationMode.FUNCTION) {
            job.setMapperClass(PairsMapperFunctionAggregation.class);
        } else {
            job.setMapperClass(PairsMapperClassAggregation.class);
        }

        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMinInputSplitSize(job, minSplitSize);
        CombineTextInputFormat.setMaxInputSplitSize(job, maxSplitSize);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);
        job.addCacheFile(top50Path.toUri());

        job.setNumReduceTasks(1);

        long startNanos = System.nanoTime();
        boolean success = job.waitForCompletion(true);
        long endNanos = System.nanoTime();
        if (!success) {
            throw new IOException("Task2 pairs job failed for mode=" + mode + ", d=" + distance);
        }

        return new RunStats(
                mode,
                distance,
                (endNanos - startNanos) / 1_000_000L,
                toMillis(job, RuntimeCounters.MAP_SETUP_NANOS),
                toMillis(job, RuntimeCounters.MAP_MAPFUNC_NANOS),
                toMillis(job, RuntimeCounters.MAP_CLEANUP_NANOS),
                toMillis(job, RuntimeCounters.REDUCE_SETUP_NANOS),
                toMillis(job, RuntimeCounters.REDUCE_REDUCEFUNC_NANOS),
                toMillis(job, RuntimeCounters.REDUCE_CLEANUP_NANOS),
                outputPath);
    }

    private static void writeRuntimeReport(
            FileSystem fs,
            Path outputBasePath,
            List<RunStats> allRuns,
            int targetSplits,
            long totalInputBytes,
            long minSplitSize,
            long maxSplitSize,
            int[] distances) throws IOException {

        Path reportPath = new Path(outputBasePath, "task2_runtime_comparison.txt");
        if (fs.exists(reportPath)) {
            fs.delete(reportPath, false);
        }

        FSDataOutputStream stream = fs.create(reportPath, true);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8));
        try {
            writer.write("task=task2_pairs");
            writer.newLine();
            writer.write("distances=" + Arrays.toString(distances));
            writer.newLine();
            writer.write("target_splits=" + targetSplits);
            writer.newLine();
            writer.write("total_input_bytes=" + totalInputBytes);
            writer.newLine();
            writer.write("min_split_size_bytes=" + minSplitSize);
            writer.newLine();
            writer.write("max_split_size_bytes=" + maxSplitSize);
            writer.newLine();
            writer.newLine();
            writer.write("mode\td\truntime_ms\tmap_setup_ms\tmap_mapFunction_ms\tmap_cleanup_ms\treduce_setup_ms\treduce_reduceFunction_ms\treduce_cleanup_ms\toutput_path");
            writer.newLine();
            for (RunStats run : allRuns) {
                writer.write(run.mode.name().toLowerCase());
                writer.write('\t');
                writer.write(String.valueOf(run.distance));
                writer.write('\t');
                writer.write(String.valueOf(run.runtimeMs));
                writer.write('\t');
                writer.write(String.valueOf(run.mapSetupMs));
                writer.write('\t');
                writer.write(String.valueOf(run.mapMapMs));
                writer.write('\t');
                writer.write(String.valueOf(run.mapCleanupMs));
                writer.write('\t');
                writer.write(String.valueOf(run.reduceSetupMs));
                writer.write('\t');
                writer.write(String.valueOf(run.reduceReduceMs));
                writer.write('\t');
                writer.write(String.valueOf(run.reduceCleanupMs));
                writer.write('\t');
                writer.write(run.outputPath.toString());
                writer.newLine();
            }
        } finally {
            writer.close();
        }
    }

    private static int[] parseDistances(String csv) {
        String[] parts = csv.split(",");
        int[] distances = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            distances[i] = Integer.parseInt(parts[i].trim());
            if (distances[i] <= 0) {
                throw new IllegalArgumentException("All distances must be positive integers.");
            }
        }
        return distances;
    }

    private static long toMillis(Job job, RuntimeCounters counter) throws IOException {
        Counter value = job.getCounters().findCounter(counter);
        return value.getValue() / 1_000_000L;
    }
}
