package com.thinkingmaze.multilayer;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

import com.thinkingmaze.nn.layers.factory.MyLayerFactories;

public class MyMultiLayerNetwork extends MultiLayerNetwork{

	public MyMultiLayerNetwork(MultiLayerConfiguration conf) {
		super(conf);
		// TODO Auto-generated constructor stub
	}

	/**
     * Initialize
     */
    public void init() {
        if (layerWiseConfigurations == null || layers == null)
            intializeConfigurations();
        if (initCalled)
            return;

        if (getnLayers() < 1)
            throw new IllegalStateException("Unable to createComplex network neuralNets; number specified is less than 1");

        if (this.layers == null || this.layers[0] == null) {
            if (this.layers == null)
                this.layers = new Layer[getnLayers()];

            // construct multi-layer
            for (int i = 0; i < getnLayers(); i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                layers[i] = MyLayerFactories.getFactory(conf).create(conf, getListeners(), i);
                layerMap.put(conf.getLayer().getLayerName(), layers[i]);
            }
            initCalled = true;
            initMask();
        }

        //Set parameters in MultiLayerNetwork.defaultConfiguration for later use in BaseOptimizer.setupSearchState() etc
        //Keyed as per backprop()
        defaultConfiguration.clearVariables();
        for( int i=0; i<layers.length; i++ ){
            for( String s : layers[i].conf().variables() ){
                defaultConfiguration.addVariable(i+"_"+s);
            }
        }

        //all params are views
        if(getLayerWiseConfigurations().isRedistributeParams())
            reDistributeParams(false);
    }
    
    private void initMask() {
        setMask(Nd4j.ones(1, pack().length()));
    }
}
