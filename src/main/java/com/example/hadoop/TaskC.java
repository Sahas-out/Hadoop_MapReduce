package com.example.hadoop;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

/**
 * TaskC: MapReduce program using STRIPE ALGORITHM to construct co-occurrence
 * matrix.
 * The stripe algorithm emits stripes (windows of co-occurring words) as values
 * with the first word as key.
 * This is more efficient than the pairs approach as it reduces intermediate
 * key-value pairs.
 * Runs for d = {1, 2, 3, 4} and reports runtime for each.
 */
public class TaskC {

    /**
     * StripeMapper - Uses stripe algorithm to find co-occurring words
     * Emits (word, stripe) pairs where stripe contains all words co-occurring with
     * word
     */
    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

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
            List<String> frequentTokens = new ArrayList<String>();
            for (String token : tokens) {
                if (!token.isEmpty() && frequentWords.contains(token)) {
                    frequentTokens.add(token);
                }
            }

            // Generate stripes: for each word, emit all co-occurring words within
            // distance d
            for (int i = 0; i < frequentTokens.size(); i++) {
                String currentWord = frequentTokens.get(i);

                // Build stripe: collect all words within distance d
                Map<String, Integer> stripe = new HashMap<String, Integer>();

                // Look both backwards and forwards within distance d
                for (int j = Math.max(0, i - distance); j <= Math.min(frequentTokens.size() - 1,
                        i + distance); j++) {
                    if (i != j) { // Exclude the word itself
                        String coWord = frequentTokens.get(j);
                        stripe.put(coWord, stripe.getOrDefault(coWord, 0) + 1);
                    }
                }

                // Convert stripe to MapWritable and emit
                if (!stripe.isEmpty()) {
                    MapWritable stripeMap = new MapWritable();
                    for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                        stripeMap.put(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                    }
                    context.write(new Text(currentWord), stripeMap);
                }
            }
        }
    }

    /**
     * StripeReducer - Merges stripes from all mappers
     * Aggregates co-occurrence counts for each word
     */
    public static class StripeReducer extends Reducer<Text, MapWritable, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            // Aggregate all stripes for this word
            Map<String, Integer> aggregatedStripe = new HashMap<String, Integer>();

            for (MapWritable stripeMap : values) {
                Iterator<Writable> iter = stripeMap.values().iterator();
                for (Writable keyWritable : stripeMap.keySet()) {
                    Text coWord = (Text) keyWritable;
                    IntWritable count = (IntWritable) stripeMap.get(coWord);
                    aggregatedStripe.put(coWord.toString(),
                            aggregatedStripe.getOrDefault(coWord.toString(), 0) + count.get());
                }
            }

            // Format output: word \t co-word:count,co-word:count,...
            if (!aggregatedStripe.isEmpty()) {
                StringBuilder stripe = new StringBuilder();
                boolean first = true;

                // Sort by co-word for consistency
                List<String> sortedWords = new ArrayList<>(aggregatedStripe.keySet());
                Collections.sort(sortedWords);

                for (String coWord : sortedWords) {
                    if (!first) {
                        stripe.append(",");
                    }
                    stripe.append(coWord).append(":").append(aggregatedStripe.get(coWord));
                    first = false;
                }

                context.write(key, new Text(stripe.toString()));
            }
        }
    }

    /**
     * Runs MapReduce job for stripe-based co-occurrence with given distance
     */
    private static boolean runStripeJob(String inputPath, String frequentWordsPath, String outputPath, int distance)
            throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("task.distance", distance);

        Job job = Job.getInstance(conf, "stripe-cooccurrence-d" + distance);
        job.setJarByClass(TaskC.class);
        job.setMapperClass(StripeMapper.class);
        job.setReducerClass(StripeReducer.class);

        // Add frequent words file to distributed cache
        job.addCacheFile(new Path(frequentWordsPath).toUri());

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        return job.waitForCompletion(true);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: TaskC <input_path> <frequent_words_file> <output_base_path>");
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
            System.out.println("\n========== STRIPE ALGORITHM CO-OCCURRENCE MATRIX ==========");
            System.out.println("Loaded frequent words file with " + lineCount + " entries\n");
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
            System.out.println("Running stripe algorithm for d=" + d);
            System.out.println("========================================");

            long startTime = System.currentTimeMillis();
            boolean success = runStripeJob(inputPath, frequentWordsPath, outputPath, d);
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
        System.out.println("RUNTIME SUMMARY - STRIPE ALGORITHM");
        System.out.println("========================================");
        for (int i = 0; i < distances.length; i++) {
            int d = distances[i];
            long runtime = runtimes[i];
            long seconds = runtime / 1000;
            long milliseconds = runtime % 1000;
            System.out.printf("Distance d=%d: %d.%03d seconds (%d ms)\n", d, seconds, milliseconds, runtime);
        }
        System.out.println("========================================");

        System.out.println("\nTaskC (Stripe Algorithm) completed successfully!");
        System.out.println("Results available in:");
        for (int d : distances) {
            System.out.println("  - d=" + d + ": " + outputBasePath + "/d" + d);
        }

        System.exit(0);
    }
}
