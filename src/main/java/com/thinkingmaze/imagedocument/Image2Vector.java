package com.thinkingmaze.imagedocument;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.commons.io.FileUtils;
import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.CNNLFWExample;
import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class Image2Vector {
	
	private static final Logger log = LoggerFactory.getLogger(Image2Vector.class);
	private static final DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
	
	private static double vectorLength(INDArray vec){
		return pow(vec, 2).sum(1).sumNumber().doubleValue();
	}
	
	@SuppressWarnings("deprecation")
	public static void main(String[] args) throws Exception {
		
		final int width = 28;
        final int height = 28;
        final int nChannels = 3;
        int batchSize = 1;
        int epochs =10;
        int iterations = 1;
        int seed = 2234;
        int exampleLength = 100;
        int numExamplesToFetch = 18000;
        int numSamples = 1;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
    	
        System.out.println("Load data....");
    	ImageIterator iter = getShakespeareIterator(batchSize, exampleLength, numExamplesToFetch);
    	
    	System.out.println("Build model....");
        MultiLayerNetwork model = new NaverNet5(width, height, nChannels, iter.inputColumns(), seed, iterations).init();
        
        System.out.println("Train model....");
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        for (int epoch = 0; epoch < epochs; epoch++){
        	iter.reset();
        	DataSet generationInitialization = iter.next();
        	String[] samples = sampleCharactersFromNetwork(generationInitialization,model,iter,numSamples,new Random(12345));
        	System.out.println("--------------------");
			System.out.println("Training epoch " + epoch );
			System.out.println("Sampling characters from network given initialization \"" + ("") + "\"");
			for( int j=0; j<samples.length; j++ ){
				System.out.println("----- Sample " + j + " -----");
				System.out.println(samples[j]);
				System.out.println();
			}
        	model.fit(iter);
        }
	}

	private static String[] sampleCharactersFromNetwork(DataSet generationInitialization, MultiLayerNetwork model,
			ImageIterator iter, int numSamples, Random random) {
		// TODO Auto-generated method stub
		INDArray initializationInput = generationInitialization.getFeatures().getRow(1);
		int imagesToSample = 300;
		
		model.rnnClearPreviousState();
		INDArray output = model.output(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output
		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder();
		
		for( int i=0; i<imagesToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = initializationInput;
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,random);
				System.out.println(Arrays.toString(outputProbDistribution));
				
				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
			}
			
			output = model.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}
		
		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
	 * DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param exampleLength Number of characters in each text segment.
	 * @param examplesPerEpoch Number of examples we want in an 'epoch'. 
	 */
	private static ImageIterator getShakespeareIterator(int miniBatchSize, int exampleLength, int numExamplesToFetch) throws Exception{
		//The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 million characters
		//https://www.gutenberg.org/ebooks/100
		File docRoot = new File("/home/cn40661/Documents/iaprtc12/annotations_complete_eng");
		String fileLocation = "/home/cn40661/Documents/iaprtc12/images";
    	for (File docFolder : docRoot.listFiles()){
    		System.out.println("read file list : " + "[" + docFolder.getPath() + "]");
    		for(File xmlFile : docFolder.listFiles()){
    			if (xmlFile.getName().endsWith("eng") == false) continue;
    			DocumentBuilder documentBuilder = builderFactory.newDocumentBuilder();
    			Document document = documentBuilder.parse(xmlFile);
    			String imageDescription = document.getElementsByTagName("DESCRIPTION").item(0).getTextContent();
    			imageDescription = imageDescription.replaceAll("[\n]+", "");
    			String imagePath = "/home/cn40661/Documents/iaprtc12/" + document.getElementsByTagName("IMAGE").item(0).getTextContent();
    			String imageDescriptionPath = imagePath.replaceAll(".jpg", ".vec");
    			OutputStreamWriter imageDescriptionFile = new OutputStreamWriter(new FileOutputStream(new File(imageDescriptionPath)));
    			imageDescriptionFile.write(imageDescription + "\n");
    			imageDescriptionFile.close();
    		}
    	}
		
		char[] validCharacters = ImageIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
		return new ImageIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, exampleLength, numExamplesToFetch, validCharacters, new Random(12345));
	}
	
	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	private static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
