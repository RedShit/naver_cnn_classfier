package com.thinkingmaze.cifar;

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
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Image2Vector {
	
	private static final Logger log = LoggerFactory.getLogger(CNNLFWExample.class);
	
	@SuppressWarnings("deprecation")
	public static void main(String[] args) throws IOException {
		
//		Scanner sin = new Scanner(new FileInputStream(new File("D://MyEclipse//iaprtc12//labelVector.txt")));
//		
//		while(sin.hasNext()){
//			String vectorPath = sin.next().replaceAll(".jpg", ".vec");
//			OutputStreamWriter vector = new OutputStreamWriter(new FileOutputStream(new File(vectorPath)));
//			vector.write(sin.nextLine()+"\n");
//			vector.close();
//		}

		final int width = 40;
        final int height = 40;
        final int nChannels = 3;
        int outputNum = CifarLoader.NUM_LABELS;
        int batchSize = 10;
        int iterations = 5;
        int splitTrainNum = (int) (batchSize*.8);
        int seed = 2234;
        int listenerFreq = iterations/5;
        
        log.info("Load data....");
        DataSetIterator cifar = new CifarDataSetIterator(
        		new CifarLoader().getRecordReader(width, height, nChannels, outputNum), 
        		batchSize, width * height * nChannels, outputNum);
        
        log.info("Build model....");
        MultiLayerNetwork model = new CifarNet(width, height, nChannels, outputNum, seed, iterations).init();
        
        log.info("Train model....");
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        while(cifar.hasNext()) {
            DataSet cifarNext = cifar.next();
            cifarNext.scale();
            SplitTestAndTrain trainTest = cifarNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            DataSet trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            // System.out.println(trainInput);
            model.fit(trainInput);
        }
	}
}
