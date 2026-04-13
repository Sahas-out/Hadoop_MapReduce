package com.example.hadoop;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;

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
 * TaskA: MapReduce program to identify top 50 most frequently occurring words.
 * Uses distributed cache to load stopwords and filters them during mapping.
 * Uses pairs approach with two-phase processing:
 * Phase 1: Count word frequencies with stopword filtering
 * Phase 2: Extract top 50 most frequent words
 */
public class TaskA {

    /**
     * Phase 1: TokenizerMapper - Tokenizes input and counts word occurrences
     * Filters stopwords using distributed cache
     */
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private Set<String> stopwords = new HashSet<String>();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            // Load stopwords from distributed cache
            if (conf.getBoolean("task.use.stopwords", false)) {
                URI[] cacheFiles = context.getCacheFiles();
                if (cacheFiles != null) {
                    for (URI cacheFile : cacheFiles) {
                        parseStopwordsFile(cacheFile.getPath());
                    }
                }
            }
        }

        private void parseStopwordsFile(String filePath) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(filePath));
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim().toLowerCase();
                    if (!line.isEmpty()) {
                        stopwords.add(line);
                    }
                }
                reader.close();
            } catch (IOException ioe) {
                System.err.println("Error reading stopwords file: " + StringUtils.stringifyException(ioe));
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            // Split on non-word characters (except apostrophes)
            String[] tokens = line.split("[^\\w']+");

            for (String token : tokens) {
                // Filter empty strings and stopwords
                if (!token.isEmpty() && !stopwords.contains(token)) {
                    word.set(token);
                    context.write(word, one);
                }
            }
        }
    }

    /**
     * Phase 1: WordCountReducer - Aggregates word frequencies
     */
    public static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
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
     * Phase 2: TopKMapper - Maps (word, count) to (count, word) for sorting
     * Also filters to keep only top 50 in mapper's local context
     */
    public static class TopKMapper extends Mapper<Object, Text, IntWritable, Text> {

        private final static int TOP_K = 50;
        // TreeMap to maintain top 50 words (sorted by count)
        private TreeMap<Integer, String> topWords = new TreeMap<Integer, String>();

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            if (parts.length == 2) {
                String word = parts[0];
                int count = Integer.parseInt(parts[1]);

                // Add to TreeMap (maintains sorted order)
                topWords.put(count, word);

                // Keep only top 50
                if (topWords.size() > TOP_K) {
                    topWords.remove(topWords.firstKey());
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Emit all top-50 words collected by this mapper
            IntWritable count = new IntWritable();
            Text word = new Text();

            // TreeMap is sorted in ascending order, so iterate in reverse
            for (Integer frequency : topWords.descendingKeySet()) {
                count.set(frequency);
                word.set(topWords.get(frequency));
                context.write(count, word);
            }
        }
    }

    /**
     * Phase 2: TopKReducer - Selects final top 50 words
     */
    public static class TopKReducer extends Reducer<IntWritable, Text, Text, IntWritable> {

        private final static int TOP_K = 50;
        // TreeMap to maintain final top 50 words (sorted by count, descending)
        private TreeMap<Integer, String> topWords = new TreeMap<Integer, String>();

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text val : values) {
                String word = val.toString();
                int count = key.get();

                // Add to TreeMap
                topWords.put(count, word);

                // Keep only top 50
                if (topWords.size() > TOP_K) {
                    topWords.remove(topWords.firstKey());
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Emit final top 50 words in descending order of frequency
            Text word = new Text();
            IntWritable count = new IntWritable();

            int rank = 1;
            for (Integer frequency : topWords.descendingKeySet()) {
                count.set(frequency);
                word.set(rank + ". " + topWords.get(frequency));
                context.write(word, count);
                rank++;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println(
                    "Usage: TaskA <input_path> <intermediate_output> <final_output> [--stopwords <stopwords_file>]");
            System.exit(1);
        }

        String inputPath = args[0];
        String intermediateOutput = args[1];
        String finalOutput = args[2];
        String stopwordsFile = null;

        // Parse optional stopwords file parameter
        for (int i = 3; i < args.length; i++) {
            if ("--stopwords".equals(args[i]) && i + 1 < args.length) {
                stopwordsFile = args[++i];
            }
        }

        // ====== PHASE 1: Count word frequencies ======
        Configuration conf1 = new Configuration();
        if (stopwordsFile != null) {
            conf1.setBoolean("task.use.stopwords", true);
        }

        Job job1 = Job.getInstance(conf1, "word-count");
        job1.setJarByClass(TaskA.class);
        job1.setMapperClass(TokenizerMapper.class);
        job1.setReducerClass(WordCountReducer.class);

        // Use CombineTextInputFormat for handling many small files efficiently
        job1.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job1, 134217728); // 128MB -
        // combines small files

        // Add stopwords file to distributed cache
        if (stopwordsFile != null) {
            job1.addCacheFile(new Path(stopwordsFile).toUri());
        }

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(intermediateOutput));

        boolean phase1Success = job1.waitForCompletion(true);
        if (!phase1Success) {
            System.out.println("Phase 1 (Word Count) failed!");
            System.exit(1);
        }

        // ====== PHASE 2: Extract top 50 words ======
        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "top-50-words");
        job2.setJarByClass(TaskA.class);
        job2.setMapperClass(TopKMapper.class);
        job2.setReducerClass(TopKReducer.class);
        job2.setNumReduceTasks(1); // Single reducer for final top-50

        job2.setMapOutputKeyClass(IntWritable.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job2, new Path(intermediateOutput));
        FileOutputFormat.setOutputPath(job2, new Path(finalOutput));

        boolean phase2Success = job2.waitForCompletion(true);
        if (!phase2Success) {
            System.out.println("Phase 2 (Top 50 Selection) failed!");
            System.exit(1);
        }

        System.out.println("TaskA completed successfully!");
        System.out.println("Results available in: " + finalOutput);
        System.exit(0);
    }
}
