package parta;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
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


public class DocumentFrequency extends Configured implements Tool {

    // -------------------------------------------------------------------------
    // Mapper: emits (stemmed_term, doc_id) for each unique term per line
    // -------------------------------------------------------------------------
    public static class DFMapper extends Mapper<Object, Text, Text, Text> {

        private final Set<String> stopwords = new HashSet<>();
        private final PorterStemmer stemmer = new PorterStemmer();
        private final Text outTerm = new Text();
        private final Text outDocId = new Text();

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
                        String w = line.trim().toLowerCase();
                        if (!w.isEmpty()) stopwords.add(w);
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

            // Document ID = filename without extension.
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
            String filename = path.getName();
            String docId = filename.replaceAll("\\.[^.]+$", "");
            outDocId.set(docId);

            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z]+");

            Set<String> seenThisLine = new HashSet<>();

            for (String token : tokens) {
                if (!isValidToken(token)) continue;
                if (stopwords.contains(token)) continue;

                String stemmed = stemmer.stem(token);
                if (!isValidToken(stemmed)) continue;
                if (stopwords.contains(stemmed)) continue;

                if (seenThisLine.add(stemmed)) {
                    outTerm.set(stemmed);
                    context.write(outTerm, outDocId);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Reducer: counts distinct document IDs for each term  ->  DF value
    // -------------------------------------------------------------------------
    public static class DFReducer extends Reducer<Text, Text, Text, IntWritable> {

        private final IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Set<String> distinctDocs = new HashSet<>();
            for (Text val : values) {
                distinctDocs.add(val.toString());
            }
            result.set(distinctDocs.size());
            context.write(key, result);
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: hadoop jar problem2.jar parta.DocumentFrequency <input> <output> <hdfs-stopwords-path>");
            return 1;
        }

        Configuration conf = getConf();

        Job job = Job.getInstance(conf, "document_frequency");
        job.setJarByClass(DocumentFrequency.class);

        job.addCacheFile(new Path(args[2]).toUri());

        job.setMapperClass(DFMapper.class);
        job.setReducerClass(DFReducer.class);

        // CombineTextInputFormat: merges small files into 64 MB splits.
        // Reduces 10,000 map tasks to ~15-20, eliminating JVM startup overhead.
        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 67108864); // 64 MB

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setNumReduceTasks(1);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new DocumentFrequency(), args));
    }
}
