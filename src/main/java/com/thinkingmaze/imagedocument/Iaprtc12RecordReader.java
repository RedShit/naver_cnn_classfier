package com.thinkingmaze.imagedocument;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import javax.imageio.ImageIO;

import org.apache.commons.io.FileUtils;
import org.canova.api.conf.Configuration;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.image.recordreader.BaseImageRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Iaprtc12RecordReader extends BaseImageRecordReader{
	
	private Map<Character,Integer> charToIdxMap;
	
	public Iaprtc12RecordReader() {
        super();
    }

    public Iaprtc12RecordReader(int width, int height, int channels, List<String> labels) {
        super(width, height, channels, labels);
    }

    public Iaprtc12RecordReader(int width, int height, int channels, boolean appendLabel, List<String> labels) {
        super(width, height, channels, appendLabel, labels);
    }

    public Iaprtc12RecordReader(int width, int height, int channels) {
        super(width, height, channels, false);
    }

    public Iaprtc12RecordReader(int width, int height, int channels, boolean appendLabel) {
        super(width, height, channels, appendLabel);
    }

    public Iaprtc12RecordReader(int width, int height, List<String> labels) {
        super(width, height, 1, labels);
    }

    public Iaprtc12RecordReader(int width, int height, boolean appendLabel, List<String> labels) {
        super(width, height, 1, appendLabel, labels);
    }

    public Iaprtc12RecordReader(int width, int height) {
        super(width, height, 1, false);
    }

    public Iaprtc12RecordReader(int width, int height, boolean appendLabel) {
        super(width, height, 1, appendLabel);
    }


    public Iaprtc12RecordReader(int width, int height, int channels, boolean appendLabel, String pattern, Map<Character,Integer> charToIdxMap) {
    	super(width, height, channels, appendLabel, pattern, 0);
    	this.charToIdxMap = charToIdxMap;
    }
    
    @Override
    public Collection<Writable> next() {
        if(iter != null) {
            Collection<Writable> ret = new ArrayList<>();
            File image = (File) iter.next();
            String vectorPath = image.getPath().replaceAll(".jpg", ".vec");
            String descriptionPath = image.getPath().replaceAll(".jpg", ".des");
            currentFile = image;
            if(image.isDirectory())
                return next();

            try {
                BufferedImage bimg = ImageIO.read(image);
                INDArray row = imageLoader.asRowVector(bimg);
                ret = RecordConverter.toRecord(row);
            	Scanner sin = new Scanner(new FileInputStream(new File(vectorPath)));
            	while(sin.hasNext()){
            		ret.add(new DoubleWritable(sin.nextDouble()));
            	}
            	sin = new Scanner(new FileInputStream(new File(descriptionPath)));
            	String description = sin.nextLine();
            	for(char c : description.toCharArray()){
            		if (charToIdxMap.keySet().contains(c) == false) continue;
            		ret.add(new IntWritable(charToIdxMap.get(c)));
            	}
            	ret.add(new IntWritable(charToIdxMap.get('\n')));
            } catch (Exception e) {
                e.printStackTrace();
            }
            return ret;
        }
        else if(record != null) {
            hitImage = true;
            return record;
        }
        throw new IllegalStateException("No more elements");
    }
    
}
