package com.example.hadoop;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

/**
 * TaskD2: Stripe approach WITH LOCAL AGGREGATION at COMBINER LEVEL
 * Uses explicit Combiner class (runs on map output) for local aggregation
 * Reduces intermediate key-value pairs with combiner function
 * Compares with TaskD2_StripeWithAggregation (in-mapper aggregation)
 * Runs for d = {1, 2, 3, 4} and reports runtime
 */
public class TaskD2_StripeWithCombiner {

    /**
     * StripeMapper - Standard stripe mapper (no local aggregation)
     * Emits stripes immediately
     */
    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> frequentWords = new HashSet<String>();
        private int distance = 1;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            distance = conf.getInt("task.distance", 1);

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

            List<String> frequentTokens = new ArrayList<String>();
            for (String token : tokens) {
                if (!token.isEmpty() && frequentWords.contains(token)) {
                    frequentTokens.add(token);
                }
            }

            // Generate stripes and emit immediately - combiner will aggregate
            for (int i = 0; i < frequentTokens.size(); i++) {
                String currentWord = frequentTokens.get(i);

                Map<String, Integer> stripe = new HashMap<String, Integer>();

                for (int j = Math.max(0, i - distance); j <= Math.min(frequentTokens.size() - 1,
                        i + distance); j++) {
                    if (i != j) {
                        String coWord = frequentTokens.get(j);
                        stripe.put(coWord, stripe.getOrDefault(coWord, 0) + 1);
                    }
                }

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
     * StripeCombiner - Aggregates stripes at map-function level
     * Runs on mapper output before sending to reducer
     */
    public static class StripeCombiner extends Reducer<Text, MapWritable, Text, MapWritable> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            Map<String, Integer> aggregatedStripe = new HashMap<String, Integer>();

            for (MapWritable stripeMap : values) {
                for (Writable keyWritable : stripeMap.keySet()) {
                    Text coWord = (Text) keyWritable;
                    IntWritable count = (IntWritable) stripeMap.get(coWord);
                    aggregatedStripe.put(coWord.toString(),
                            aggregatedStripe.getOrDefault(coWord.toString(), 0) + count.get());
                }
            }

            if (!aggregatedStripe.isEmpty()) {
                MapWritable combinedStripe = new MapWritable();
                for (Map.Entry<String, Integer> entry : aggregatedStripe.entrySet()) {
                    combinedStripe.put(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                }
                context.write(key, combinedStripe);
            }
        }
    }

    /**
     * StripeReducer - Final aggregation
     */
    public static class StripeReducer extends Reducer<Text, MapWritable, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            Map<String, Integer> aggregatedStripe = new HashMap<String, Integer>();

            for (MapWritable stripeMap : values) {
                for (Writable keyWritable : stripeMap.keySet()) {
                    Text coWord = (Text) keyWritable;
                    IntWritable count = (IntWritable) stripeMap.get(coWord);
                    aggregatedStripe.put(coWord.toString(),
                            aggregatedStripe.getOrDefault(coWord.toString(), 0) + count.get());
                }
            }

            if (!aggregatedStripe.isEmpty()) {
                StringBuilder stripe = new StringBuilder();
                boolean first = true;

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

    private static boolean runStripeJob(String inputPath, String frequentWordsPath, String outputPath, int distance)
            throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("task.distance", distance);

        Job job = Job.getInstance(conf, "stripe-combiner-d" + distance);
        job.setJarByClass(TaskD2_StripeWithCombiner.class);
        job.setMapperClass(StripeMapper.class);
        job.setCombinerClass(StripeCombiner.class); // Enable combiner for local aggregation
        job.setReducerClass(StripeReducer.class);

        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 134217728);

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
            System.err
                    .println("Usage: TaskD2_StripeWithCombiner <input_path> <frequent_words_file> <output_base_path>");
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
            System.out.println("\n========== STRIPE WITH COMBINER AGGREGATION ==========");
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
            System.out.println("Stripe + Combiner Aggregation for d=" + d);
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

        System.out.println("\n========================================");
        System.out.println("RUNTIME SUMMARY - STRIPE + COMBINER AGGREGATION");
        System.out.println("========================================");
        for (int i = 0; i < distances.length; i++) {
            int d = distances[i];
            long runtime = runtimes[i];
            long seconds = runtime / 1000;
            long milliseconds = runtime % 1000;
            System.out.printf("Distance d=%d: %d.%03d seconds (%d ms)\n", d, seconds, milliseconds, runtime);
        }
        System.out.println("========================================");

        System.out.println("\nTaskD2 (Combiner) completed successfully!");
        System.out.println("Results available in:");
        for (int d : distances) {
            System.out.println("  - d=" + d + ": " + outputBasePath + "/d" + d);
        }

        System.exit(0);
    }
}
