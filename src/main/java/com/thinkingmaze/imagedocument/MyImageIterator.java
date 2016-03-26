package com.thinkingmaze.imagedocument;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.split.LimitFileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/** A very simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file to start the sequence and
 * (optionally) scanning backwards to a new line (to ensure we don't start half way through a word
 * for example).<br>
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class MyImageIterator implements DataSetIterator {
	private static final long serialVersionUID = -7287833919126626356L;
	private char[] validCharacters;
	private Map<Character,Integer> charToIdxMap;
	private int exampleLength;
	private int numExamplesToFetch;
	private int miniBatchSize;
	private int examplesSoFar = 0;
	private int examplesPerEpoch;
	private Random rng;
	private final int numCharacters;
	
	private RecordReader recordReader;
	private ArrayList<List<Writable>> currList;
	private int width = 28;
	private int height = 28;
	private int channels = 3;
	private boolean appendLabel = false;
	private final String regexPattern = ".[#]+";
	public static final String[] ALLOWED_FORMATS = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};
	
	
	public MyImageIterator(String path, int miniBatchSize, int exampleSize, int numExamplesToFetch, int examplesPerEpoch ) throws IOException {
		this(path,Charset.defaultCharset(),miniBatchSize,exampleSize,numExamplesToFetch,examplesPerEpoch,getDefaultCharacterSet(), new Random());
	}
	
	/**
	 * @param textFilePath Path to text file to use for generating samples
	 * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
	 * @param miniBatchSize Number of examples per mini-batch
	 * @param exampleLength Number of characters in each input/output vector
	 * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
	 * @param rng Random number generator, for repeatability if required
	 * @param alwaysStartAtNewLine if true, scan backwards until we find a new line character (up to MAX_SCAN_LENGTH in case
	 *  of no new line characters, to avoid scanning entire file)
	 * @throws IOException If text file cannot  be loaded
	 */
	public MyImageIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength, int numExamplesToFetch,
			int examplesPerEpoch, char[] validCharacters, Random rng ) throws IOException {
		if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
		if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
		this.validCharacters = validCharacters;
		this.exampleLength = exampleLength;
		this.miniBatchSize = miniBatchSize;
		this.examplesPerEpoch = examplesPerEpoch;
		this.numExamplesToFetch = numExamplesToFetch;
		this.examplesSoFar = 0;
		this.rng = rng;
		
		//Store valid characters is a map for later use in vectorization
		charToIdxMap = new HashMap<>();
		for( int i=0; i<validCharacters.length; i++ ) charToIdxMap.put(validCharacters[i], i);
		numCharacters = validCharacters.length;
		
		recordReader = new Iaprtc12RecordReader(width, height, channels, appendLabel, regexPattern, charToIdxMap);
		try {
			File fullDir = new File(textFilePath);
			InputSplit split = new LimitFileSplit(fullDir, ALLOWED_FORMATS, exampleLength, regexPattern, rng);
            recordReader.initialize(split);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
		
		this.currList = new ArrayList<List<Writable>>();
		while(recordReader.hasNext()){
			currList.add((List<Writable>) recordReader.next());
		}
	}
	
	/** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
	public static char[] getMinimalCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c='a'; c<='z'; c++) validChars.add(c);
		for(char c='A'; c<='Z'; c++) validChars.add(c);
		for(char c='0'; c<='9'; c++) validChars.add(c);
		char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
		for( char c : temp ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}
	
	/** As per getMinimalCharacterSet(), but with a few extra characters */
	public static char[] getDefaultCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c : getMinimalCharacterSet() ) validChars.add(c);
		char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
				'\\', '|', '<', '>'};
		for( char c : additionalChars ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}
	
	public char convertIndexToCharacter( int idx ){
		return validCharacters[idx];
	}
	
	public int convertCharacterToIndex( char c ){
		return charToIdxMap.get(c);
	}
	
	public char getRandomCharacter(){
		return validCharacters[(int) (rng.nextDouble()*validCharacters.length)];
	}

	public boolean hasNext() {
		return examplesSoFar + miniBatchSize <= examplesPerEpoch;
	}


	public DataSet next(int num) {
		if( examplesSoFar + miniBatchSize > examplesPerEpoch ) throw new NoSuchElementException();
		//Allocate space:
		
		int imageV = width * height * channels;
		int featureLength = 77;
		int preChar = 0;
		INDArray input = Nd4j.zeros(num,numExamplesToFetch, imageV);
		INDArray labels = Nd4j.zeros(num,numExamplesToFetch, numCharacters);
		
		for (int i = 0; i < num; i++){
			int idx = (int) (rng.nextDouble()* currList.size()) ;
			List<Writable> imageAndDescription = currList.get(idx);
			for (int j = 0; j<numExamplesToFetch && j+featureLength+imageV<imageAndDescription.size(); j++){
				for (int k = 0; k < imageV; k++){
					Writable current = imageAndDescription.get(k);
					input.putScalar(new int[]{j, k}, current.toDouble());
				}
				Writable current = imageAndDescription.get(featureLength+imageV+j);
				labels.putScalar(new int[]{j, current.toInt()}, 1.0f);
//				if (j == 0){
//					for (int k = 0; k < featureLength; k++){
//						Writable current = imageAndDescription.get(k+imageV);
//						input.putScalar(new int[]{i, k, j}, current.toDouble());
//					}
//					Writable current = imageAndDescription.get(featureLength+imageV+j);
//					labels.putScalar(new int[]{i, current.toInt(), j}, 1);
//					preChar = current.toInt();
//				}
//				else{
//					input.putScalar(new int[]{i, preChar, j}, 1.0f);
//					Writable current = imageAndDescription.get(featureLength+imageV+j);
//					labels.putScalar(new int[]{i, current.toInt(), j}, 1.0f);
//					preChar = current.toInt();
//				}
			}
		}
		examplesSoFar += miniBatchSize;
		DataSet ret = new DataSet(input,labels);
		
		return ret;
	}


	public int inputColumns() {
		return numCharacters;
	}

	public int totalOutcomes() {
		return numCharacters;
	}

	public void reset() {
		examplesSoFar = 0;
	}

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return examplesSoFar;
	}

	public int numExamples() {
		return numExamplesToFetch;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DataSet next() {
		return next(miniBatchSize);
	}

	@Override
	public int totalExamples() {
		// TODO Auto-generated method stub
		return 20000;
	}

}