package com.example.hadoop;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineFileRecordReader;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;

public class CombinedWholeFileInputFormat extends CombineFileInputFormat<Text, Text> {

    @Override
    public RecordReader<Text, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        return new CombineFileRecordReader<Text, Text>(
                (CombineFileSplit) split,
                context,
                WholeFileInCombineRecordReader.class);
    }

    public static class WholeFileInCombineRecordReader extends RecordReader<Text, Text> {
        private final CombineFileSplit split;
        private final TaskAttemptContext context;
        private final int fileIndex;

        private boolean processed = false;
        private Text currentKey;
        private Text currentValue;

        public WholeFileInCombineRecordReader(CombineFileSplit split, TaskAttemptContext context, Integer fileIndex) {
            this.split = split;
            this.context = context;
            this.fileIndex = fileIndex;
        }

        @Override
        public void initialize(InputSplit inputSplit, TaskAttemptContext taskAttemptContext) {
            // No-op: split, context, and file index are provided in constructor by CombineFileRecordReader.
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            if (processed) {
                return false;
            }

            Path filePath = split.getPath(fileIndex);
            long fileLength = split.getLength(fileIndex);
            if (fileLength > Integer.MAX_VALUE) {
                throw new IOException("File is too large for in-memory whole-file read: " + filePath);
            }

            byte[] contents = new byte[(int) fileLength];
            FileSystem fs = filePath.getFileSystem(context.getConfiguration());
            FSDataInputStream inputStream = fs.open(filePath);
            try {
                inputStream.readFully(0, contents);
            } finally {
                inputStream.close();
            }

            currentKey = new Text(filePath.toString());
            currentValue = new Text(new String(contents, StandardCharsets.UTF_8));
            processed = true;
            return true;
        }

        @Override
        public Text getCurrentKey() {
            return currentKey;
        }

        @Override
        public Text getCurrentValue() {
            return currentValue;
        }

        @Override
        public float getProgress() {
            return processed ? 1.0f : 0.0f;
        }

        @Override
        public void close() {
            // no resources to close
        }
    }
}
