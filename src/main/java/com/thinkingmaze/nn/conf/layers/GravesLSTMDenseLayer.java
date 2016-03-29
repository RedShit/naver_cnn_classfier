package com.thinkingmaze.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

/**LSTM Dense layer: fully connected feed forward layer trainable by backprop for LSTM.
 */


import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class GravesLSTMDenseLayer extends FeedForwardLayer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7868686949682153168L;

	private GravesLSTMDenseLayer(Builder builder) {
    	super(builder);
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")
        public GravesLSTMDenseLayer build() {
            return new GravesLSTMDenseLayer(this);
        }
    }
}
