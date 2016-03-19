package com.thinkingmaze.cifar;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


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
public class CifarNet {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;

    public CifarNet(int height, int width, int channels, int outputNum, long seed, int iterations) {
        // TODO consider ways to make this adaptable to other problems not just imagenet
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
        // TODO batch size set to 128 for ImageNet based on paper - base it on memory bandwidth
    }

    public MultiLayerConfiguration conf() {
        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        // TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
		        .seed(seed)
		        .iterations(iterations)
		        .activation("relu")
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
		        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		        .learningRate(0.01)
		        .momentum(0.9)
		        .regularization(true)
		        .updater(Updater.ADAGRAD)
		        .useDropConnect(true)
		        .list(6)
		        .layer(0, new ConvolutionLayer.Builder(4, 4)
		                .name("cnn1")
		                .nIn(channels)
		                .stride(1, 1)
		                .nOut(20)
		                .build())
		        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
		                .name("pool1")
		                .build())
		        .layer(2, new ConvolutionLayer.Builder(3, 3)
		                .name("cnn2")
		                .stride(1,1)
		                .nOut(40)
		                .build())
		        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
		                .name("pool2")
		                .build())
		        .layer(2, new ConvolutionLayer.Builder(3, 3)
		                .name("cnn3")
		                .stride(1,1)
		                .nOut(60)
		                .build())
		        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
		                .name("pool3")
		                .build())
		        .layer(2, new ConvolutionLayer.Builder(2, 2)
		                .name("cnn3")
		                .stride(1,1)
		                .nOut(80)
		                .build())
		        .layer(4, new DenseLayer.Builder()
		                .name("ffn1")
		                .nOut(160)
		                .dropOut(0.5)
		                .build())
		        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
		                .nOut(outputNum)
		                .activation("softmax")
		                .build())
		        .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(conf, height, width, channels);
        
        return conf.build();
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork model = new MultiLayerNetwork(conf());
        model.init();
        return model;

    }

}
