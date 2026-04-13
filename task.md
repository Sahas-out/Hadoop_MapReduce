
## Problem . CO-OCCURRING WORD MATRIX GENERATION: PAIRS & STRIPES AND LOCAL AGGREGATION 14 Points
1. Develop a MapReduce program aimed at identifying the top 50 most frequently occur-
ring words from the provided Wikipedia dump located (Wikipedia-EN-20120601_ART-
ICLES). Utilize the pairs approach for this task. Exclude stop words(stopwords.txt provided) while computing the frequency of occurrence. Utilize the distributed caching option within Hadoop(for loading stopwords), to eliminate all stop-words in the mapper.
2. Write another MapReduce program, employing the pairs approach, to construct the
co-occurring word matrix based solely on the frequent words identified in the previous question.
Consider a word distance of d when determining co-occurring pairs. Report the runtime for
d = {1, 2, 3, 4}.
3. Develop a stripe algorithm to create the co-occurring word matrix. Again, report the
runtime for d = {1, 2, 3, 4}.
4. Additionally, implement local aggregations separately at the Map-class level and the
Map-function level, and compare their performance. Conduct this comparison for parts (b)
and (c). Once again, the runtime must be reported for d = {1, 2, 3, 4}.

## Context
1. Wikipedia-EN-20120601_ART has 10000 files with file sizes only being around 16KB.
2. My system has 16 cores usage up to 12 is completely ok.
3. Need to run experiments with different d so give a placeholder to configure in code
4. this is a maven project you can refer to pom.xml for details

## Guidelines
1. As you can see there are a lot of small files to process what i believe is it would create a lot of splits and thus to optimize we should have around 10-12 splits as around 12 cores so use combineTextInputFormat.
2. Also change the mapred-conf.xml file accordingly so that we can use run 12 task concurrently.
3. dont refer to TaskA.java TaskB.java and TASKC.java in the codespace 
4. create three files one for task 1 in problem statement one for task 2 and one for task 3 and have three mapreduce classes 1. without local aggregation 2. with local aggregation at function level 3. with local aggregation at class level
5. we also need atleast runtime metric so add print statements for that.
6. we can cache the top50 words for tasks 3 and task 4
