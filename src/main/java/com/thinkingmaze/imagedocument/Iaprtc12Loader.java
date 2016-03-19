package com.thinkingmaze.imagedocument;

import java.io.File;
import java.io.IOException;
import java.net.URI;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;

public class Iaprtc12Loader extends BaseImageLoader{
	
	public final static int NUM_IMAGES = 100000;
    public final static int NUM_LABELS = 100;
    
	private final boolean appendLabel = false;
	private final String regexPattern = ".[#]+";
	
	protected String localDir = "Documents/iaprtc12/images";
	protected String subLocalDir = "images";
	protected File fullDir = new File(BASE_DIR, localDir);
	protected int numExamples = NUM_IMAGES;
    protected int numLabels = NUM_LABELS;
    
    public RecordReader getRecordReader(int width, int height, int channels, int numExamples) {
    	this.numExamples = numExamples;
    	return getRecordReader(width, height, channels);
    }
	
	public RecordReader getRecordReader(int width, int height, int channels) {
		RecordReader recordReader = new Iaprtc12RecordReader(width, height, channels, appendLabel, regexPattern);
		try {
			InputSplit split = new LimitFileSplit(fullDir, ALLOWED_FORMATS, numExamples, numLabels, regexPattern, rng);
            recordReader.initialize(split);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return recordReader;
	}
}
