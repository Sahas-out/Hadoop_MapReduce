package com.example.hadoop;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

/**
 * TaskB: MapReduce program to construct co-occurrence matrix for top frequent
 * words.
 * Uses pairs approach to find co-occurring words within distance d.
 * Runs for d = {1, 2, 3, 4} and reports runtime for each.
 */
public class TaskB {

    /**
     * CoOccurrenceMapper - Finds co-occurring word pairs within distance d
     * Loads top frequent words from distributed cache
     */
    public static class CoOccurrenceMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Set<String> frequentWords = new HashSet<String>();
        private int distance = 1;

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
                    // Parse output from TaskA: "rank. word\tcount"
                    String[] parts = line.split("\t");
                    if (parts.length > 0) {
                        // Extract word from "rank. word" format
                        String wordPart = parts[0].trim();
                        String word = wordPart.replaceAll("^\\d+\\.\\s+", "").toLowerCase();
                        if (!word.isEmpty()) {
                            frequentWords.add(word);
                        }
                    }
                }
                reader.close();
                System.out.println("Loaded " + frequentWords.size() + " frequent words for distance d=" + distance);
            } catch (IOException ioe) {
                System.err.println("Error reading frequent words file: " + StringUtils.stringifyException(ioe));
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            // Tokenize: split on non-word characters (except apostrophes)
            String[] tokens = line.split("[^\\w']+");

            // Filter tokens to keep only frequent words
            java.util.List<String> frequentTokens = new java.util.ArrayList<String>();
            for (String token : tokens) {
                if (!token.isEmpty() && frequentWords.contains(token)) {
                    frequentTokens.add(token);
                }
            }

            // Generate co-occurrence pairs within distance d
            for (int i = 0; i < frequentTokens.size(); i++) {
                String word1 = frequentTokens.get(i);
                // Look ahead within distance d
                for (int j = i + 1; j <= Math.min(i + distance, frequentTokens.size() - 1); j++) {
                    String word2 = frequentTokens.get(j);
                    // Create ordered pair to avoid duplicates (word1 < word2 lexicographically)
                    if (word1.compareTo(word2) < 0) {
                        context.write(new Text(word1 + "\t" + word2), one);
                    } else {
                        context.write(new Text(word2 + "\t" + word1), one);
                    }
                }
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

    /**
     * Runs MapReduce job for co-occurrence matrix with given distance
     */
    private static boolean runCoOccurrenceJob(String inputPath, String frequentWordsPath,
            String outputPath, int distance) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("task.distance", distance);

        Job job = Job.getInstance(conf, "co-occurrence-d" + distance);
        job.setJarByClass(TaskB.class);
        job.setMapperClass(CoOccurrenceMapper.class);
        job.setReducerClass(CoOccurrenceReducer.class);

        // Add frequent words file to distributed cache
        job.addCacheFile(new Path(frequentWordsPath).toUri());

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        return job.waitForCompletion(true);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: TaskB <input_path> <frequent_words_file> <output_base_path>");
            System.err.println("  input_path: Path to original wiki documents");
            System.err.println("  frequent_words_file: Output from TaskA (top 50 words)");
            System.err.println("  output_base_path: Base path for outputs (d1, d2, d3, d4 subdirectories)");
            System.exit(1);
        }

        String inputPath = args[0];
        String frequentWordsPath = args[1];
        String outputBasePath = args[2];

        // Verify that frequent words file exists
        try {
            BufferedReader reader = new BufferedReader(new FileReader(frequentWordsPath));
            int lineCount = 0;
            while (reader.readLine() != null) {
                lineCount++;
            }
            reader.close();
            System.out.println("\nLoaded frequent words file with " + lineCount + " entries\n");
        } catch (IOException e) {
            System.err.println("Error reading frequent words file: " + frequentWordsPath);
            System.exit(1);
        }

        // Run jobs for each distance value
        int[] distances = { 1, 2, 3, 4 };
        long[] runtimes = new long[4];

        for (int i = 0; i < distances.length; i++) {
            int d = distances[i];
            String outputPath = outputBasePath + "/d" + d;

            System.out.println("\n========================================");
            System.out.println("Running co-occurrence analysis for d=" + d);
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

        // Print runtime summary
        System.out.println("\n========================================");
        System.out.println("RUNTIME SUMMARY FOR CO-OCCURRENCE MATRIX");
        System.out.println("========================================");
        for (int i = 0; i < distances.length; i++) {
            int d = distances[i];
            long runtime = runtimes[i];
            long seconds = runtime / 1000;
            long milliseconds = runtime % 1000;
            System.out.printf("Distance d=%d: %d.%03d seconds (%d ms)\n", d, seconds, milliseconds, runtime);
        }
        System.out.println("========================================");

        System.out.println("\nTaskB completed successfully!");
        System.out.println("Results available in:");
        for (int d : distances) {
            System.out.println("  - d=" + d + ": " + outputBasePath + "/d" + d);
        }

        System.exit(0);
    }
}
