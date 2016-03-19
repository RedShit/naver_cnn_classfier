package com.thinkingmaze.imagedocument;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.deeplearning4j.examples.nlp.paragraphvectors.ParagraphVectorsTextExample;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.w3c.dom.Document;

// //home//cn40661//Documents//iaprtc12//annotations_complete_eng
// //home//cn40661//Documents//iaprtc12//sentences.txt

public class Paragraph2Vector {
	
	private static final Logger log = LoggerFactory.getLogger(Paragraph2Vector.class);
	private static final DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();

	
	private static class Rank implements Comparable<Rank>{
		private double similarity = 0;
		private String doc = "";
		
		public double getSimilarity() {
			return similarity;
		}
		public void setSimilarity(double similarity) {
			this.similarity = similarity;
		}
		public String getDoc() {
			return doc;
		}
		public void setDoc(String doc) {
			this.doc = doc;
		}
		
		Rank() {}
		Rank(double similarity, String doc){
			this.setSimilarity(similarity);
			this.setDoc(doc);
		}
		
		@Override
		public int compareTo(Rank o) {
			// TODO Auto-generated method stub
			if (this.getSimilarity() < o.getSimilarity())
				return 1;
			if (this.getSimilarity() > o.getSimilarity())
				return -1;
			return 0;
		}
		
		
	}
	
	private static String doubles2String(double[] ds){
		String ret = "";
		for (double d : ds){
			ret = ret + String.valueOf(d) + " ";
		}
		return ret;
	}
    public static void main(String[] args) throws Exception {
    	Map doc2Vector = new HashMap();
    	File docRoot = new File("/home/cn40661/Documents/iaprtc12/annotations_complete_eng");
    	OutputStreamWriter SentencesFile = new OutputStreamWriter(new FileOutputStream(
    			new File("/home/cn40661/git/dl4j-0.4-examples/target/classes/raw_sentences.txt")));
    	int idx = 0;
    	for (File docFolder : docRoot.listFiles()){
    		System.out.println("read file list : " + "[" + docFolder.getPath() + "]");
    		for(File xmlFile : docFolder.listFiles()){
    			if (xmlFile.getName().endsWith("eng") == false) continue;
    			DocumentBuilder documentBuilder = builderFactory.newDocumentBuilder();
    			Document document = documentBuilder.parse(xmlFile);
    			String imageDescription = document.getElementsByTagName("DESCRIPTION").item(0).getTextContent();
    			imageDescription = imageDescription.replaceAll("[\n]+", "");
    			String imagePath = "/home/cn40661/Documents/iaprtc12/" + document.getElementsByTagName("IMAGE").item(0).getTextContent();
    			doc2Vector.put("DOC_"+idx, imagePath);
    			SentencesFile.write(imageDescription+"\n");
    			++ idx;
    		}
    	}
    	File docs = new File("/home/cn40661/Documents/iaprtc12/raw_sentences.txt");
    	Scanner sin = new Scanner(new FileInputStream(docs));
		while(sin.hasNextLine()){
			SentencesFile.write(sin.nextLine()+'\n');
		}
		sin.close();
    	SentencesFile.close();
    	ClassPathResource resource = new ClassPathResource("/raw_sentences.txt");
        File file = resource.getFile();
        System.out.println(file.getPath());
        SentenceIterator iter = new BasicLineIterator(file);

        InMemoryLookupCache cache = new InMemoryLookupCache();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        /*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.

              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
        */
        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(30)
                .epochs(20)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();
        
        log.info("train end ... \n");
        
        OutputStreamWriter labelVectorFile = new OutputStreamWriter(new FileOutputStream(
    			new File("/home/cn40661/Documents/iaprtc12/labelVector.txt")));
        for (int i = 0; i < idx; ++i){
        	labelVectorFile.write((String)doc2Vector.get("DOC_"+i) + " " + doubles2String(vec.getWordVector("DOC_"+i)) + "\n");
        }
        labelVectorFile.close();
        
        log.info("model end ... \n");
        
        /*
            In training corpus we have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space

            line 10: two men are sitting on the roof of a red bus on a gravel raod with a densely wooded slope in the background.
            line 88: a red fruit on a brown branch; green plants in the background;
            line 217: a man is sitting in a stable and is milking a brown cow; there is a second cow behind him;
            line 319: a woman is hugging a man in the middle of a flat, yellowish brown sandy desert;
            line 473: a woman is sitting on a dark green floor, is leaning at a white wall and is reading a postcard;  .

            this is special sentence, that has nothing common with previous sentences
            line 505: brown petroglyphs in a grey rock; .
         */
//    	OutputStreamWriter resultFile = new OutputStreamWriter(new FileOutputStream(
//    			new File("//home//cn40661//Documents//iaprtc12//result.txt")));
//    	
//    	for(Object doc : doc2sentence.keySet().toArray()){
//    		List<Rank> re = new ArrayList<Rank>();
//    		for(Object tDoc : doc2sentence.keySet().toArray()){
//    			double similarity = vec.similarity((String) doc, (String) tDoc);
//    			re.add(new Rank(similarity, (String)tDoc));
//        	}
//    		Collections.sort(re);
//    		for(int i = 0; i < 5; ++i){
//    			resultFile.write((String)doc2sentence.get(re.get(i).getDoc())+'\n');
//    		}
//    		resultFile.write("\n\n\n\n\n\n");
//    	}
//    	resultFile.close();
    }
}
