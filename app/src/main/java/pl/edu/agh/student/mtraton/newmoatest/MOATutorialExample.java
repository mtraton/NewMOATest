package pl.edu.agh.student.mtraton.newmoatest;

/**
 * Created by Rael on 15.01.2016.
 */

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.generators.RandomRBFGenerator;
import weka.core.Instance;


public class MOATutorialExample {

    public static void main(String[] args) {
        int numInstances =10000;
        Classifier learner=new HoeffdingTree();
        RandomRBFGenerator stream=new RandomRBFGenerator();
        stream.prepareForUse();

        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        int numberSamplesCorrect=0;
        int numberSamples=0;
        boolean isTesting=true;
        while(stream.hasMoreInstances()&&numberSamples<numInstances){
         Instance trainInst=stream.nextInstance();
            if(isTesting){
                if(learner.correctlyClassifies(trainInst)){
                    numberSamplesCorrect++;
                }
            }
            numberSamples++;
            learner.trainOnInstance(trainInst);
        }
        double accuracy=100.0*(double)numberSamplesCorrect/(double)numberSamples;
        System.out.println(numberSamples+" instances processed with"+accuracy+"% accuracy");
    }
}

