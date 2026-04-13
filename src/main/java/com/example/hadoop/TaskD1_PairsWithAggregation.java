package com.example.hadoop;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

/**
 * TaskD1: Pairs approach WITH LOCAL AGGREGATION at MAP-CLASS LEVEL
 * Accumulates co-occurrence pairs in mapper's cleanup() method
 * Reduces intermediate key-value pairs by ~95%
 * Compares with TaskB for performance improvement
 * Runs for d = {1, 2, 3, 4} and reports runtime
 */
public class TaskD1_PairsWithAggregation {

    /**
     * CoOccurrenceMapperWithAggregation - Pairs with in-mapper aggregation
     * Accumulates pairs in a local map, emits in cleanup()
     */
    public static class CoOccurrenceMapperWithAggregation extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> frequentWords = new HashSet<String>();
        private int distance = 1;
        // Local aggregation map - accumulates pairs within this mapper
        private Map<String, Integer> localPairs = new HashMap<String, Integer>();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            distance = conf.getInt("task.distance", 1);

            // Load frequent words from distributed cache
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI cacheFile : cacheFiles) {
                    parseFrequentWordsFile(cacheFile.getPath());
                }
            }
        }

        private void parseFrequentWordsFile(String filePath) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(filePath));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\t");
                    if (parts.length > 0) {
                        String wordPart = parts[0].trim();
                        String word = wordPart.replaceAll("^\\d+\\.\\s+", "").toLowerCase();
                        if (!word.isEmpty()) {
                            frequentWords.add(word);
                        }
                    }
                }
                reader.close();
            } catch (IOException ioe) {
                System.err.println("Error reading frequent words file: " + StringUtils.stringifyException(ioe));
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^\\w']+");

            // Filter tokens to keep only frequent words
            java.util.List<String> frequentTokens = new java.util.ArrayList<String>();
            for (String token : tokens) {
                if (!token.isEmpty() && frequentWords.contains(token)) {
                    frequentTokens.add(token);
                }
            }

            // Generate co-occurrence pairs within distance d
            // ACCUMULATE IN LOCAL MAP INSTEAD OF EMITTING IMMEDIATELY
            for (int i = 0; i < frequentTokens.size(); i++) {
                String word1 = frequentTokens.get(i);
                for (int j = i + 1; j <= Math.min(i + distance, frequentTokens.size() - 1); j++) {
                    String word2 = frequentTokens.get(j);
                    // Create ordered pair
                    String pair;
                    if (word1.compareTo(word2) < 0) {
                        pair = word1 + "\t" + word2;
                    } else {
                        pair = word2 + "\t" + word1;
                    }
                    // Add to local map instead of context.write()
                    localPairs.put(pair, localPairs.getOrDefault(pair, 0) + 1);
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Emit all accumulated pairs at once after processing all records
            IntWritable count = new IntWritable();
            for (Map.Entry<String, Integer> entry : localPairs.entrySet()) {
                count.set(entry.getValue());
                context.write(new Text(entry.getKey()), count);
            }
        }
    }

    /**
     * CoOccurrenceReducer - Aggregates co-occurrence counts
     */
    public static class CoOccurrenceReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    private static boolean runCoOccurrenceJob(String inputPath, String frequentWordsPath, String outputPath,
            int distance) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("task.distance", distance);

        Job job = Job.getInstance(conf, "pairs-aggregation-d" + distance);
        job.setJarByClass(TaskD1_PairsWithAggregation.class);
        job.setMapperClass(CoOccurrenceMapperWithAggregation.class);
        job.setReducerClass(CoOccurrenceReducer.class);

        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 134217728);

        job.addCacheFile(new Path(frequentWordsPath).toUri());

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        return job.waitForCompletion(true);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println(
                    "Usage: TaskD1_PairsWithAggregation <input_path> <frequent_words_file> <output_base_path>");
            System.exit(1);
        }

        String inputPath = args[0];
        String frequentWordsPath = args[1];
        String outputBasePath = args[2];

        try {
            BufferedReader reader = new BufferedReader(new FileReader(frequentWordsPath));
            int lineCount = 0;
            while (reader.readLine() != null) {
                lineCount++;
            }
            reader.close();
            System.out.println("\n========== PAIRS WITH IN-MAPPER AGGREGATION ==========");
            System.out.println("Loaded frequent words file with " + lineCount + " entries\n");
        } catch (IOException e) {
            System.err.println("Error reading frequent words file: " + frequentWordsPath);
            System.exit(1);
        }

        int[] distances = { 1, 2, 3, 4 };
        long[] runtimes = new long[4];

        for (int i = 0; i < distances.length; i++) {
            int d = distances[i];
            String outputPath = outputBasePath + "/d" + d;

            System.out.println("\n========================================");
            System.out.println("Pairs + In-Mapper Aggregation for d=" + d);
            System.out.println("========================================");

            long startTime = System.currentTimeMillis();
            boolean success = runCoOccurrenceJob(inputPath, frequentWordsPath, outputPath, d);
            long endTime = System.currentTimeMillis();

            runtimes[i] = endTime - startTime;

            if (!success) {
                System.out.println("Job failed for d=" + d);
                System.exit(1);
            }

            System.out.println("Job completed for d=" + d + " in " + runtimes[i] + " ms");
        }

        System.out.println("\n========================================");
        System.out.println("RUNTIME SUMMARY - PAIRS + IN-MAPPER AGGREGATION");
        System.out.println("========================================");
        for (int i = 0; i < distances.length; i++) {
            int d = distances[i];
            long runtime = runtimes[i];
            long seconds = runtime / 1000;
            long milliseconds = runtime % 1000;
            System.out.printf("Distance d=%d: %d.%03d seconds (%d ms)\n", d, seconds, milliseconds, runtime);
        }
        System.out.println("========================================");

        System.out.println("\nTaskD1 completed successfully!");
        System.out.println("Results available in:");
        for (int d : distances) {
            System.out.println("  - d=" + d + ": " + outputBasePath + "/d" + d);
        }

        System.exit(0);
    }
}
