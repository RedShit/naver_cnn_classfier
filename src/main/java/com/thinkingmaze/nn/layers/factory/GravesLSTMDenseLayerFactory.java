package com.thinkingmaze.nn.layers.factory;

import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.layers.factory.DefaultLayerFactory;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;

public class GravesLSTMDenseLayerFactory extends DefaultLayerFactory {

    public GravesLSTMDenseLayerFactory(Class<? extends Layer> layerConfig) {
		super(layerConfig);
		// TODO Auto-generated constructor stub
	}
    
    @Override
    protected org.deeplearning4j.nn.api.Layer getInstance(NeuralNetConfiguration conf) {
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.DenseLayer)
            return new org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.AutoEncoder)
            return new org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.RBM)
            return new org.deeplearning4j.nn.layers.feedforward.rbm.RBM(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.ImageLSTM)
            return new org.deeplearning4j.nn.layers.recurrent.ImageLSTM(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.GravesLSTM)
        	return new org.deeplearning4j.nn.layers.recurrent.GravesLSTM(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.GRU )
        	return new org.deeplearning4j.nn.layers.recurrent.GRU(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.OutputLayer)
            return new org.deeplearning4j.nn.layers.OutputLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.RnnOutputLayer)
        	return new org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder)
            return new org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.RecursiveAutoEncoder(conf);   
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.ConvolutionLayer)
            return new org.deeplearning4j.nn.layers.convolution.ConvolutionLayer(conf);   
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.SubsamplingLayer)
            return new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.BatchNormalization)
            return new org.deeplearning4j.nn.layers.normalization.BatchNormalization(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.LocalResponseNormalization)
            return new org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization(conf);
        if(layerConfig instanceof com.thinkingmaze.nn.conf.layers.GravesLSTMDenseLayer)
        	return new com.thinkingmaze.nn.layers.feedforward.dense.GravesLSTMDenseLayer(conf);
        throw new RuntimeException("unknown layer type: " + layerConfig);
    }
}
