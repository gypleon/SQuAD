PATH_CORENLP="../corenlp/stanford-corenlp-full-2017-06-09/*"
# Run the server using all jars in the current directory (e.g., the CoreNLP home directory)
# java -cp "$PATH_CORENLP" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref
java -Xmx500m -cp "$PATH_CORENLP" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators tokenize -port 9000 -timeout 15000
