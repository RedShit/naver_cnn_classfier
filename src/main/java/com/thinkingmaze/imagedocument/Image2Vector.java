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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.CNNLFWExample;
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

import com.thinkingmaze.imagedocument.Iaprtc12Loader;

public class Image2Vector {
	
	private static final Logger log = LoggerFactory.getLogger(CNNLFWExample.class);
	
	private static double vectorLength(INDArray vec){
		return pow(vec, 2).sum(1).sumNumber().doubleValue();
	}
	
	@SuppressWarnings("deprecation")
	public static void main(String[] args) throws IOException {
		
//		Scanner sin = new Scanner(new FileInputStream(new File("/home/cn40661/Documents/iaprtc12/labelVector.txt")));
//		
//		while(sin.hasNext()){
//			String vectorPath = sin.next().replaceAll(".jpg", ".vec");
//			OutputStreamWriter vector = new OutputStreamWriter(new FileOutputStream(new File(vectorPath)));
//			double[] arr = new double[100];
//			int curr = 0;
//			double sum = 0;
//			while(sin.hasNextDouble()){
//				arr[curr] = (sin.nextDouble());
//				sum += arr[curr];
//				curr++;
//			}
//			for(int i = 0; i < 100; i++){
//				vector.write((arr[i]) + " ");
//			}
//			vector.write("\n");
//			vector.close();
//		}
//		sin.close();

		final int width = 28;
        final int height = 28;
        final int nChannels = 3;
        int outputNum = Iaprtc12Loader.NUM_LABELS;
        int numExamples = Iaprtc12Loader.NUM_IMAGES;
        int batchSize = 20;
        int epochs =10;
        int iterations = 3;
        int splitTrainNum = (int) (batchSize*.9);
        int seed = 2234;
        int listenerFreq = iterations/5;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        
        log.info("Load data....");
        DataSetIterator cifar = new Iaprtc12DataSetIterator(
        		new Iaprtc12Loader().getRecordReader(width, height, nChannels, numExamples), 
        		batchSize, width * height * nChannels, outputNum);
        
        log.info("Build model....");
        MultiLayerNetwork model = new LeNet5(width, height, nChannels, outputNum, seed, iterations).init();
        
        log.info("Train model....");
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        for (int epoch = 0; epoch < epochs; epoch++){
        	cifar.reset();
	        int curr = 0;
	        while(cifar.hasNext()) {
	            DataSet cifarNext = cifar.next();
	            cifarNext.scale();
	            SplitTestAndTrain trainTest = cifarNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
	            DataSet trainInput = trainTest.getTrain(); // get feature matrix and labels for training
	            testInput.add(trainTest.getTest().getFeatureMatrix());
	            testLabels.add(trainTest.getTest().getLabels());
	            model.fit(trainInput);
	            curr += 20;
	            System.out.println("train completed " + epoch + " epochs " + (curr/200.0) + "%");
	        }
        }
        
        OutputStreamWriter imageVector = new OutputStreamWriter(new FileOutputStream(new File("/home/cn40661/Documents/iaprtc12/imageVector.txt")));
        log.info("Evaluate model....");
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            for(int b = 0; b < output.shape()[0]; b++){
            	imageVector.write(output.getRow(b) + "\n");
            	imageVector.write(testLabels.get(i).getRow(b) + "\n");
            }
        }
        imageVector.close();
	}
}
