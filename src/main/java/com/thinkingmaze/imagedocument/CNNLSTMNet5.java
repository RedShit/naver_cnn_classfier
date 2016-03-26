package com.thinkingmaze.imagedocument;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * References:
 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
 * https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
 *
 * Dl4j's AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
 * and the imagenetExample code referenced.
 *
 * Model is built in dl4j based on available functionality and notes indicate where there are gaps waiting for enhancements.
 * Created by nyghtowl on 9/11/15.
 *
 * Bias initialization in the paper is 1 in certain layers but 0.1 in the imagenetExample code
 * Weight distribution uses 0.1 std for all layers in the paper but 0.005 in the dense layers in the imagenetExample code
 *
 */
@Deprecated
public class CNNLSTMNet5 {

    private int height;
    private int width;
    private int channels = 3;
    private long seed = 123;
    private int iterations = 90;
    private int lstmLayerSize = 200;
    private int nOut = 100;

    public CNNLSTMNet5(int height, int width, int channels, int nOut, long seed, int iterations) {
        // TODO consider ways to make this adaptable to other problems not just imagenet
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.nOut = nOut;
        this.seed = seed;
        this.iterations = iterations;
        // TODO batch size set to 128 for ImageNet based on paper - base it on memory bandwidth
    }

    public MultiLayerConfiguration conf() {

        // TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
    	MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.learningRate(0.1)
				.rmsDecay(0.95)
				.seed(12345)
				.regularization(true)
				.l2(0.001)
				.list(7)
				.layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20).dropOut(0.5)
                        .activation("sigmoid")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation("sigmoid")
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nOut(100)
                        .activation("tanh")
                        .build())
				.layer(4, new GravesLSTM.Builder().nIn(100).nOut(lstmLayerSize)
						.updater(Updater.RMSPROP)
						.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08)).build())
				.layer(5, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
						.updater(Updater.RMSPROP)
						.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08)).build())
				.layer(6, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
						.updater(Updater.RMSPROP)
						.nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08)).build())
				.pretrain(false).backprop(true);
				
        new ConvolutionLayerSetup(conf, height, width, channels);
        
        return conf.build();
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork model = new MultiLayerNetwork(conf());
        model.init();
        return model;

    }
}
