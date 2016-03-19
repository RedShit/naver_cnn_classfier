package com.thinkingmaze.cifar;

import java.io.File;
import java.io.IOException;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;

public class CifarLoader extends BaseImageLoader{
	
	public final static int NUM_IMAGES = 19999;
    public final static int NUM_LABELS = 100;
    
	private final boolean appendLabel = true;
	private final String regexPattern = ".[#]+";
	
	protected String localDir = "D:\\MyEclipse\\iaprtc12\\iaprtc12";
	protected File fullDir = new File(localDir);
	protected int numExamples = NUM_IMAGES;
    protected int numLabels = NUM_LABELS;
    
    public RecordReader getRecordReader(int width, int height, int channels, int numExamples) {
    	this.numExamples = numExamples;
    	return getRecordReader(width, height, channels);
    }
	
	public RecordReader getRecordReader(int width, int height, int channels) {
		RecordReader recordReader = new CifarRecordReader(width, height, channels, appendLabel, regexPattern);
		try {
            recordReader.initialize(new LimitFileSplit(fullDir, ALLOWED_FORMATS, numExamples, numLabels, regexPattern, rng));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return recordReader;
	}
}
