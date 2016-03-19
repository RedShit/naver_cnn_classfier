package com.thinkingmaze.imagedocument;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.SelfWritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

public class Iaprtc12DataSetIterator implements DataSetIterator{

	private RecordReader recordReader;
    private WritableConverter converter;
    private int batchSize = 10;
    private int labelIndex = -1;
    private int numPossibleLabels = -1;
    private boolean overshot = false;
    private Iterator<Collection<Writable>> sequenceIter;
    private DataSet last;
    private boolean useCurrent = false;
    private boolean regression = false;
    private DataSetPreProcessor preProcessor;

    /**
     * Use the record reader and batch size; no labels
     * @param recordReader the record reader to use
     * @param batchSize the batch size of the data
     */
    public Iaprtc12DataSetIterator(RecordReader recordReader, int batchSize) {
        this(recordReader, new SelfWritableConverter(), batchSize, -1, -1);
    }

    /**
     * Main constructor
     * @param recordReader the recorder to use for the dataset
     * @param batchSize the batch size
     * @param labelIndex the index of the label to use
     * @param numPossibleLabels the number of posible
     */
    public Iaprtc12DataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels);
    }

    /**
     *
     * @param recordReader
     */
    public Iaprtc12DataSetIterator(RecordReader recordReader) {
        this(recordReader, new SelfWritableConverter());
    }


    /**
     * Invoke the CifarDataSetIterator with a batch size of 10
     * @param recordReader the recordreader to use
     * @param labelIndex the index of the label
     * @param numPossibleLabels the number of possible labels for classification
     *
     */
    public Iaprtc12DataSetIterator(RecordReader recordReader, int labelIndex, int numPossibleLabels) {
        this(recordReader, new SelfWritableConverter(), 10, labelIndex, numPossibleLabels);
    }


    /**
     *
     * @param recordReader
     * @param converter
     * @param batchSize
     * @param labelIndex
     * @param numPossibleLabels
     * @param regression
     */
    public Iaprtc12DataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels,boolean regression) {
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
    }
    public Iaprtc12DataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels) {
        this(recordReader,converter,batchSize,labelIndex,numPossibleLabels,false);
    }

    public Iaprtc12DataSetIterator(RecordReader recordReader, WritableConverter converter) {
        this(recordReader, converter, 10, -1, -1);
    }


    public Iaprtc12DataSetIterator(RecordReader recordReader, WritableConverter converter, int labelIndex, int numPossibleLabels) {
        this(recordReader, converter, 10, labelIndex, numPossibleLabels);
    }


    @Override
    public DataSet next(int num) {
        if(useCurrent) {
            useCurrent = false;
            if(preProcessor != null) preProcessor.preProcess(last);
            return last;
        }

        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            if (!hasNext())
                break;
            if (recordReader instanceof SequenceRecordReader) {
                if(sequenceIter == null || !sequenceIter.hasNext()) {
                    Collection<Collection<Writable>> sequenceRecord = ((SequenceRecordReader) recordReader).sequenceRecord();
                    sequenceIter = sequenceRecord.iterator();
                }
                Collection<Writable> record = sequenceIter.next();
                dataSets.add(getDataSet(record));
            }

            else {
                Collection<Writable> record = recordReader.next();
                dataSets.add(getDataSet(record));
            }
        }
        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for (DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }

        if(inputs.isEmpty()) {
            overshot = true;
            return last;
        }
        DataSet ret =  new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
        last = ret;
        if(preProcessor != null) preProcessor.preProcess(ret);
        return ret;
    }


    private DataSet getDataSet(Collection<Writable> record) {
        List<Writable> currList;
        if (record instanceof List)
            currList = (List<Writable>) record;
        else
            currList = new ArrayList<>(record);

        //allow people to specify label index as -1 and infer the last possible label
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = record.size() - numPossibleLabels;
        }
        INDArray label = Nd4j.create(numPossibleLabels);
        INDArray featureVector = Nd4j.create(labelIndex >= 0 ? currList.size()-numPossibleLabels : currList.size());
        int featureCount = 0;
        for (int j = 0; j < currList.size(); j++) {
            Writable current = currList.get(j);
            if (current.toString().isEmpty())
                continue;
            if (labelIndex >= 0 && j >= labelIndex) {
                if (converter != null)
                    try {
                        current = converter.convert(current);
                    } catch (WritableConverterException e) {
                        e.printStackTrace();
                    }
                if (numPossibleLabels < 1)
                    throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
                if (regression) {
                    label = Nd4j.scalar(current.toDouble());
                } else {
                    double curr = current.toDouble();
                    label.putScalar(j-labelIndex, curr);
                }
            } else {
                featureVector.putScalar(featureCount++, current.toDouble());
            }
        }

        return new DataSet(featureVector, labelIndex >= 0 ? label : featureVector);
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        if(last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        }
        else
            return last.numInputs();

    }

    @Override
    public int totalOutcomes() {
        if(last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        }
        else
            return last.numOutcomes();
    }

    @Override
    public void reset() {
        if (recordReader instanceof RecordReader)
            recordReader.reset();
        else if (recordReader instanceof SequenceRecordReader)
            throw new UnsupportedOperationException("Reset not supported for SequenceRecordReader type.");
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }



    @Override
    public boolean hasNext() {
        return recordReader.hasNext() || overshot;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels(){
        return recordReader.getLabels();
    }
	
}
