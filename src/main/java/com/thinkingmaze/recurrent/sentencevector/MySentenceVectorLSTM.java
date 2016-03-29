package com.thinkingmaze.recurrent.sentencevector;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.thinkingmaze.multilayer.MyMultiLayerNetwork;
import com.thinkingmaze.nn.conf.layers.GravesLSTMDenseLayer;

public class MySentenceVectorLSTM {
	private int height;
    private int width;
    private int channels = 3;
    private long seed = 123;
    private int iterations = 90;
    private int lstmLayerSize = 200;
    private int sentenceVectorSize = 200;
    private int sentenceNumber = 10000;
    private int nOut = 100;

    public MySentenceVectorLSTM(int height, int width, int channels, int nOut, long seed, 
    		int iterations, int sentenceVectorSize, int sentenceNumber) {
        // TODO consider ways to make this adaptable to other problems not just imagenet
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.nOut = nOut;
        this.seed = seed;
        this.iterations = iterations;
        this.sentenceVectorSize = sentenceVectorSize;
        this.sentenceNumber = sentenceNumber;
    }
    public MultiLayerConfiguration conf() throws InstantiationException, IllegalAccessException {
        // TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.learningRate(0.1)
				.rmsDecay(0.95)
				.seed(12345)
				.regularization(true)
				.l2(0.001)
				.list(4)
				.layer(0, new GravesLSTMDenseLayer.Builder().nIn(sentenceNumber).nOut(sentenceVectorSize)
                        .weightInit(WeightInit.XAVIER)
                        .activation("tanh")
                        .build())
				.layer(1, new GravesLSTM.Builder().nIn(sentenceVectorSize).nOut(lstmLayerSize)
						.updater(Updater.RMSPROP)
						.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08)).build())
				.layer(2, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
						.updater(Updater.RMSPROP)
						.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08)).build())
				.layer(3, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
						.updater(Updater.RMSPROP)
						.nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08)).build())
				.pretrain(false).backprop(true)
				.build();
        
        return conf;
    }

    public MultiLayerNetwork init() throws InstantiationException, IllegalAccessException{
        MultiLayerNetwork model = new MyMultiLayerNetwork(conf());
        model.init();
        return model;

    }
}
