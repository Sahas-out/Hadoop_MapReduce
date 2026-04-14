package partb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import opennlp.tools.stemmer.PorterStemmer;

/**
 * Problem 2b: TF-IDF scoring over Wikipedia articles.
 *
 * Uses CombineTextInputFormat (same as 2a) to combine 10,000 small files into
 * large splits, reducing map tasks from 10,000 to ~15-20 for fast execution.
 *
 * Outputs a TSV file with schema:  ID<tab>TERM<tab>SCORE
 */
public class TFIDFScorer extends Configured implements Tool {

    private static final String PAIR_SEP = "|";
    private static final String KV_SEP   = ":";

    // -------------------------------------------------------------------------
    // Mapper — stripes algorithm with in-mapper combining
    // -------------------------------------------------------------------------
    public static class StripeMapper extends Mapper<Object, Text, Text, Text> {

        private final Map<String, Integer> dfMap = new HashMap<>();
        private final Map<String, Map<String, Integer>> docStripes = new HashMap<>();

        private final PorterStemmer stemmer = new PorterStemmer();
        private final Text outKey   = new Text();
        private final Text outValue = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            URI[] cacheFiles = Job.getInstance(conf).getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) return;

            for (URI uri : cacheFiles) {
                Path filePath = new Path(uri.getPath());
                String fileName = filePath.getName();
                try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\t");
                        if (parts.length == 2) {
                            try {
                                dfMap.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                            } catch (NumberFormatException ignored) { }
                        }
                    }
                }
            }
        }

        private boolean isValidToken(String token) {
            if (token == null) return false;
            token = token.trim();
            if (token.length() <= 2) return false;
            if (!token.matches("[a-z]+")) return false;
            if (token.matches("^(i|ii|iii|iv|v|vi|vii|viii|ix|x)+$")) return false;
            return true;
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            // CombineTextInputFormat sets mapreduce.map.input.file per file within the split.
            Configuration conf = context.getConfiguration();
            String inputFile = conf.get("mapreduce.map.input.file");
            if (inputFile == null) {
                inputFile = conf.get("map.input.file");
            }
            Path path;
            if (inputFile != null) {
                path = new Path(inputFile);
            } else {
                org.apache.hadoop.mapreduce.InputSplit split = context.getInputSplit();
                if (!(split instanceof FileSplit)) {
                    throw new IOException("Unsupported split type: " + split.getClass());
                }
                path = ((FileSplit) split).getPath();
            }
            String docId = path.getName().replaceAll("\\.[^.]+$", "");

            Map<String, Integer> stripe = docStripes.computeIfAbsent(docId, k -> new HashMap<>());

            String line    = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z]+");

            for (String token : tokens) {
                if (!isValidToken(token)) continue;
                String stemmed = stemmer.stem(token);
                if (isValidToken(stemmed) && dfMap.containsKey(stemmed)) {
                    stripe.merge(stemmed, 1, Integer::sum);
                }
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<String, Map<String, Integer>> docEntry : docStripes.entrySet()) {
                outKey.set(docEntry.getKey());

                StringBuilder sb = new StringBuilder();
                for (Map.Entry<String, Integer> termEntry : docEntry.getValue().entrySet()) {
                    if (sb.length() > 0) sb.append(PAIR_SEP);
                    sb.append(termEntry.getKey()).append(KV_SEP).append(termEntry.getValue());
                }
                outValue.set(sb.toString());
                context.write(outKey, outValue);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Reducer — merges stripes, computes TF-IDF, emits ID<tab>TERM<tab>SCORE
    // -------------------------------------------------------------------------
    public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {

        private final Map<String, Integer> dfMap = new HashMap<>();
        private final Text outValue = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            URI[] cacheFiles = Job.getInstance(conf).getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) return;

            for (URI uri : cacheFiles) {
                Path filePath = new Path(uri.getPath());
                String fileName = filePath.getName();
                try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\t");
                        if (parts.length == 2) {
                            try {
                                dfMap.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                            } catch (NumberFormatException ignored) { }
                        }
                    }
                }
            }
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            Map<String, Integer> termCounts = new HashMap<>();
            for (Text stripeText : values) {
                String[] pairs = stripeText.toString().split("\\" + PAIR_SEP);
                for (String pair : pairs) {
                    int sep = pair.indexOf(KV_SEP);
                    if (sep < 0) continue;
                    String term = pair.substring(0, sep);
                    try {
                        int count = Integer.parseInt(pair.substring(sep + 1));
                        termCounts.merge(term, count, Integer::sum);
                    } catch (NumberFormatException ignored) { }
                }
            }

            for (Map.Entry<String, Integer> entry : termCounts.entrySet()) {
                String term = entry.getKey();
                int    tf   = entry.getValue();
                int    df   = dfMap.getOrDefault(term, 1);

                double score = tf * Math.log(10000.0 / df + 1.0);

                outValue.set(term + "\t" + String.format("%.6f", score));
                context.write(key, outValue);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Tool entry point
    // -------------------------------------------------------------------------
    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println(
                "Usage: hadoop jar problem2.jar partb.TFIDFScorer " +
                "<input-dir> <output-dir> <hdfs-df_top100-path>");
            return 1;
        }

        Configuration conf = getConf();

        Job job = Job.getInstance(conf, "tfidf_scorer");
        job.setJarByClass(TFIDFScorer.class);

        job.addCacheFile(new Path(args[2]).toUri());

        job.setMapperClass(StripeMapper.class);
        job.setReducerClass(TFIDFReducer.class);

        // CombineTextInputFormat: same strategy as 2a — merge small files into 64 MB splits.
        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 67108864); // 64 MB

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new TFIDFScorer(), args));
    }
}
