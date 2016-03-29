package com.thinkingmaze.nn.layers.factory;

import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GRU;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.ImageLSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.factory.BatchNormalizationLayerFactory;
import org.deeplearning4j.nn.layers.factory.ConvolutionLayerFactory;
import org.deeplearning4j.nn.layers.factory.DefaultLayerFactory;
import org.deeplearning4j.nn.layers.factory.GRULayerFactory;
import org.deeplearning4j.nn.layers.factory.GravesLSTMLayerFactory;
import org.deeplearning4j.nn.layers.factory.ImageLSTMLayerFactory;
import org.deeplearning4j.nn.layers.factory.LocalResponseNormalizationFactory;
import org.deeplearning4j.nn.layers.factory.PretrainLayerFactory;
import org.deeplearning4j.nn.layers.factory.RecursiveAutoEncoderLayerFactory;
import org.deeplearning4j.nn.layers.factory.SubsampleLayerFactory;

import com.thinkingmaze.nn.conf.layers.GravesLSTMDenseLayer;

public class MyLayerFactories {
	/**
     * Get the factory based on the passed in class
     * @param conf the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(NeuralNetConfiguration conf) {
        return getFactory(conf.getLayer());
    }

    /**
     * Get the factory based on the passed in class
     * @param layer the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(Layer layer) {
        Class<? extends Layer> clazz = layer.getClass();
        if(clazz.equals(ImageLSTM.class))
            return new ImageLSTMLayerFactory(ImageLSTM.class);
        else if(clazz.equals(GravesLSTM.class))
        	return new GravesLSTMLayerFactory(GravesLSTM.class);
        else if(clazz.equals(GRU.class))
        	return new GRULayerFactory(GRU.class);
        else if(RecursiveAutoEncoder.class.isAssignableFrom(clazz))
            return new RecursiveAutoEncoderLayerFactory(RecursiveAutoEncoder.class);
        else if(BasePretrainNetwork.class.isAssignableFrom(clazz))
            return new PretrainLayerFactory(clazz);
        else if(ConvolutionLayer.class.isAssignableFrom(clazz))
            return new ConvolutionLayerFactory(clazz);
        else if(SubsamplingLayer.class.isAssignableFrom(clazz))
            return new SubsampleLayerFactory(clazz);
        else if(BatchNormalization.class.isAssignableFrom(clazz))
            return new BatchNormalizationLayerFactory(clazz);
        else if(LocalResponseNormalization.class.isAssignableFrom(clazz))
            return new LocalResponseNormalizationFactory(clazz);
        else if(clazz.equals(GravesLSTMDenseLayer.class))
            return new GravesLSTMDenseLayerFactory(clazz);
        return new DefaultLayerFactory(clazz);
    }


    /**
     * Get the type for the layer factory
     * @param conf the layer factory
     * @return the type
     */
    public static org.deeplearning4j.nn.api.Layer.Type typeForFactory(NeuralNetConfiguration conf) {
        LayerFactory layerFactory = getFactory(conf);
        if(layerFactory instanceof ConvolutionLayerFactory || layerFactory instanceof SubsampleLayerFactory)
            return org.deeplearning4j.nn.api.Layer.Type.CONVOLUTIONAL;
        else if(layerFactory instanceof ImageLSTMLayerFactory || layerFactory instanceof GravesLSTMLayerFactory
        		|| layerFactory instanceof GRULayerFactory )
            return org.deeplearning4j.nn.api.Layer.Type.RECURRENT;
        else if(layerFactory instanceof RecursiveAutoEncoderLayerFactory)
            return org.deeplearning4j.nn.api.Layer.Type.RECURSIVE;
        else if(layerFactory instanceof DefaultLayerFactory || layerFactory instanceof PretrainLayerFactory)
            return org.deeplearning4j.nn.api.Layer.Type.FEED_FORWARD;

        throw new IllegalArgumentException("Unknown layer type");
    }
}
