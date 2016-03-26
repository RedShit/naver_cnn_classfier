package com.thinkingmaze.imagedocument;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.Random;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

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

public class Image2Text {
	
	private static final Logger log = LoggerFactory.getLogger(Image2Text.class);
	private static final DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
	
	@SuppressWarnings("deprecation")
	public static void main(String[] args) throws Exception {
		
		final int width = 1;
        final int height = 1;
        final int nChannels = 1;
        int batchSize = 32;
        int epochs = 50;
        int iterations = 1;
        int seed = 2234;
        int exampleLength = 20000;
        int examplesPerEpoch = batchSize * 1000;
        int numExamplesToFetch = 80;
        int numSamples = 1;
    	
        System.out.println("Load data....");
    	ImageIterator iter = getShakespeareIterator(batchSize, exampleLength, numExamplesToFetch, examplesPerEpoch);
    	
    	System.out.println("Build model....");
        MultiLayerNetwork model = new NaverNet5(width, height, nChannels, iter.inputColumns(), seed, iterations).init();
        
        System.out.println("Train model....");
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        
        OutputStreamWriter descriptionFile = new OutputStreamWriter(
        		new FileOutputStream(new File("D:/MyEclipse/iaprtc12/descriptionFile.txt")));
        
        for (int epoch = 0; epoch < epochs; epoch++){
        	iter.reset();
        	model.fit(iter);
        	iter.reset();
        	DataSet generationInitialization = iter.next();
        	String[] samples = sampleCharactersFromNetwork(generationInitialization,model,iter,new Random(12345));
        	
        	System.out.println("--------------------");
			System.out.println("Training epoch " + epoch );
			System.out.println("Sampling characters from network given initialization \"" + ("") + "\"");
			descriptionFile.write("--------------------\nTraining epoch " + epoch + "\n");
			
			numSamples = samples.length/2;
			for( int j=0; j<numSamples; j++ ){
				System.out.println("----- Sample " + j + " -----");
				System.out.println(samples[j]);
				System.out.println("--------------------------------------------------");
				System.out.println(samples[j+numSamples]);
				System.out.println();
				descriptionFile.write("----- Sample " + j + " -----" + "\n");
				descriptionFile.write(samples[j] + "\n");
				descriptionFile.write("--------------------------------------------------\n");
				descriptionFile.write(samples[j+numSamples] + "\n");
			}
			
        }
        descriptionFile.close();
	}

	private static String[] sampleCharactersFromNetwork(DataSet generationInitialization, MultiLayerNetwork model,
			ImageIterator iter, Random random) {
		// TODO Auto-generated method stub
		INDArray initializationInput = (INDArray) generationInitialization.getFeatures();
		INDArray initializationLabel = (INDArray) generationInitialization.getLabels();
		int imagesToSample = generationInitialization.getLabels().size(0);
		int numSamples = generationInitialization.getLabels().size(2);
		int featureLength = 77;
		
		StringBuilder[] sb = new StringBuilder[imagesToSample*2];
		for( int i=0; i<imagesToSample*2; i++ ) sb[i] = new StringBuilder();
		
		for( int i=0; i<imagesToSample; i++ ){
			INDArray input = Nd4j.zeros(1, featureLength);
			for(int j = 0; j < featureLength; j++){
				input.putScalar(new int[]{0,j}, initializationInput.getDouble(i, j, 0));
			}
			model.rnnClearPreviousState();
			INDArray output = model.rnnTimeStep(input);
			//Set up next input (single time step) by sampling from previous output
			
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,random);
				
				INDArray nextInput = Nd4j.zeros(1, featureLength);
				nextInput.putScalar(new int[]{0,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[i].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
				output = model.rnnTimeStep(nextInput);	//Do one time step of forward pass
				
				double[] labelProbDistribution = new double[iter.totalOutcomes()];
				double sum = 0;
				for( int j=0; j<labelProbDistribution.length; j++ ) {
					labelProbDistribution[j] = initializationLabel.getDouble(i, j, s);
					sum += labelProbDistribution[j];
				}
				if (sum < 0.5) continue;
				int labelCharacterIdx = sampleFromDistribution(labelProbDistribution,random);
				sb[i+imagesToSample].append(iter.convertIndexToCharacter(labelCharacterIdx));
				
			}
		}
		
		String[] out = new String[imagesToSample*2];
		for( int i=0; i<imagesToSample*2; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
	 * DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param exampleLength Number of characters in each text segment.
	 * @param examplesPerEpoch Number of examples we want in an 'epoch'. 
	 */
	private static ImageIterator getShakespeareIterator(int miniBatchSize, int exampleLength, int numExamplesToFetch, int examplesPerEpoch) throws Exception{
		//The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 million characters
		//https://www.gutenberg.org/ebooks/100
		File docRoot = new File("D:/MyEclipse/iaprtc12/annotations_complete_eng");
		String fileLocation = "D:/MyEclipse/iaprtc12/images";
    	for (File docFolder : docRoot.listFiles()){
    		System.out.println("read file list : " + "[" + docFolder.getPath() + "]");
    		for(File xmlFile : docFolder.listFiles()){
    			if (xmlFile.getName().endsWith("eng") == false) continue;
    			DocumentBuilder documentBuilder = builderFactory.newDocumentBuilder();
    			Document document = documentBuilder.parse(xmlFile);
    			String imageDescription = document.getElementsByTagName("DESCRIPTION").item(0).getTextContent();
    			imageDescription = imageDescription.replaceAll("[\n]+", "");
    			String imagePath = "D:/MyEclipse/iaprtc12/" + document.getElementsByTagName("IMAGE").item(0).getTextContent();
    			String imageDescriptionPath = imagePath.replaceAll(".jpg", ".des");
    			OutputStreamWriter imageDescriptionFile = new OutputStreamWriter(new FileOutputStream(new File(imageDescriptionPath)));
    			imageDescriptionFile.write(imageDescription + "\n");
    			imageDescriptionFile.close();
    		}
    	}

    	char[] validCharacters = ImageIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
		return new ImageIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, exampleLength, numExamplesToFetch, examplesPerEpoch, validCharacters, new Random(12345));
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
