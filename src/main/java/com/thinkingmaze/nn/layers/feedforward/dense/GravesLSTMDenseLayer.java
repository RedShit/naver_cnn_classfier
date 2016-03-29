package com.thinkingmaze.nn.layers.feedforward.dense;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.thinkingmaze.nn.layers.MyBaseLayer;

public class GravesLSTMDenseLayer extends MyBaseLayer<com.thinkingmaze.nn.conf.layers.GravesLSTMDenseLayer>{

	public GravesLSTMDenseLayer(NeuralNetConfiguration conf) {
		super(conf);
		// TODO Auto-generated constructor stub
	}
	
	public GravesLSTMDenseLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        // TODO Auto-generated constructor stub
    }

}
