import weka.classifiers.*;
import weka.core.*;

import java.util.*;
import weka.filters.supervised.instance.Resample;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.*;
import java.io.*;
import weka.classifiers.trees.j48.*;
import weka.classifiers.trees.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

class Option {
  String flag, opt;
  public Option(String flag, String opt) { this.flag = flag; this.opt = opt; }
}

public class Main {
  final static Map<String, List<String>> params = new HashMap<>();

  static Instances trainData = null, testData = null, randData = null;
  static Instance unseenData = null;
  static Classifier classifier = null;
  static Boolean isModelLoaded = false;

  Boolean isUseTestData = false;

  private static void getArgumentsList(String[] args){
    List<String> options = null;
    for (int i = 0; i < args.length; i++) {
        final String a = args[i];

        if (a.charAt(0) == '-') {
            if (a.length() < 2) {
                System.err.println("Error at argument " + a);
                return;
            }

            options = new ArrayList<>();
            params.put(a.substring(1), options);
        }
        else if (options != null) {
            options.add(a);
        }
        else {
            System.err.println("Illegal parameter usage");
            return;
        }
    }
  }

  private static Instances loadData(String path) throws Exception {
    DataSource source = new DataSource(path);
    Instances data = source.getDataSet();

    data.setClassIndex(data.numAttributes() - 1);
    return data;
  }

  private static Instance loadUnseenData(String path) throws Exception {
    return loadData(path).firstInstance();
  }

  private static Classifier loadModel(String modelType){
    if (modelType.equalsIgnoreCase("j48")) {
      return new J48();
    }

    return new Id3();
  }

  private static void testModel() throws Exception {
    if (testData != null) {
      classifier.buildClassifier(trainData);

      // evaluate classifier and print some statistics
      Evaluation evaluation = new Evaluation(trainData);
      evaluation.evaluateModel(classifier, testData);

      System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
    } else {
      if (params.get("percentage-split") != null && params.get("percentage-split").size() > 0) {
        int seed = 0;
        Random rand = new Random(seed);   // create seeded number generator
        randData = new Instances(trainData);   // create copy of original data
        randData.randomize(rand);         // randomize data with number generator

        // Percentage Split
        int splitPercentage = Integer.parseInt(params.get("percentage-split").get(0));

        int trainSize = (int) Math.round(randData.numInstances() * splitPercentage / 100);
        int testSize = randData.numInstances() - trainSize;

        Instances splitTrainData = new Instances(randData, 0, trainSize);
        Instances splitTestData = new Instances(randData, trainSize, testSize);

        classifier.buildClassifier(splitTrainData);

        Evaluation evaluation = new Evaluation(splitTrainData);
        evaluation.evaluateModel(classifier, splitTestData);

        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
        System.out.println(evaluation.toMatrixString());
      } else {
        int folds = params.get("k-fold") != null ? Integer.parseInt(params.get("k-fold").get(0)) : 10;

        classifier.buildClassifier(trainData);

        Evaluation evaluation = new Evaluation(trainData);
        evaluation.crossValidateModel(classifier, trainData, folds, new Random(1));

        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
        System.out.println(evaluation.toMatrixString());
      }
    }
  }

  private static void saveModel() throws Exception {
    // serialize model
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("myModel.model"));
    oos.writeObject(classifier);
    oos.flush();
    oos.close();
  }

  private static Classifier loadModelFromExternal() throws Exception {
    // deserialize model
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("myModel.model"));
    Classifier classifier = (Classifier) ois.readObject();
    ois.close();

    return classifier;
  }

  private static Instances resample(Instances data) {
    final Resample filter = new Resample();
    Instances filteredIns = null;
    filter.setBiasToUniformClass(1.0);

    try {
      filter.setInputFormat(data);
      filter.setNoReplacement(params.get("resample-no-replacement") != null ? Boolean.parseBoolean(params.get("resample-no-replacement").get(0)) : false);
      filter.setSampleSizePercent(params.get("resample-sample-size-percent") != null ? Integer.parseInt(params.get("resample-sample-size-percent").get(0)) : 100);
      filter.setInvertSelection(params.get("resample-invert-selection") != null ? Boolean.parseBoolean(params.get("resample-invert-selection").get(0)) : false);
      filteredIns = Filter.useFilter(data, filter);
    } catch (Exception e) {
      e.printStackTrace();
    }

    return filteredIns;
  }

  private static Instances removeAttribute(Instances inst, String[] indices) throws Exception {
    Remove remove = new Remove();

    for (String index : indices) {
      remove.setAttributeIndices(index);
    }

    remove.setInvertSelection(false);
    remove.setInputFormat(inst);
    return Filter.useFilter(inst, remove);
  }

  private static void classifyUnseenData(Classifier classifier, Instance unseenData) throws Exception {
    System.out.println("Class : " + classifier.classifyInstance(unseenData));
  }

  public static void main(String[] args) throws Exception {
    getArgumentsList(args);

    Boolean isRemoveAttribute = params.get("remove-attribute") != null ? true : false;
    Boolean isPercentageSplit = params.get("percentage-split") != null ? true : false;
    Boolean isKFold = params.get("k-fold") != null ? true : false;
    Boolean isModel = params.get("model") != null ? true : false;
    Boolean isResample = params.get("resample") != null ? true : false;
    Boolean isLoadModel = params.get("load-model") != null ? true : false;
    Boolean isSaveModel = params.get("save-model") != null ? true : false;
    Boolean isUnseenDataArff = params.get("unseen-data") != null ? true : false;
    Boolean isTrainDataArff = params.get("train-data") != null ? true : false;
    Boolean isTestDataArff = params.get("test-data") != null ? true : false;

    if (isTrainDataArff) {
      trainData = loadData(params.get("train-data").get(0));
    }

    if (isTestDataArff) {
      testData = loadData(params.get("test-data").get(0));
    }

    if (isRemoveAttribute) {
      if (trainData != null) {
        trainData = removeAttribute(trainData, params.get("remove-attribute").get(0).toString().split(","));
      }

      if (testData != null) {
        testData = removeAttribute(testData, params.get("remove-attribute").get(0).toString().split(","));
      }
    }

    if (isResample) {
      if (trainData != null) trainData = resample(trainData);
      if (testData != null) testData = resample(testData);
    }

    if (isUnseenDataArff) {
      unseenData = loadUnseenData(params.get("unseen-data").get(0));

      if (isLoadModel) {
        classifier = loadModelFromExternal();
      } else {
        classifier = loadModel(params.get("model") != null ? params.get("model").get(0) : "id3");
        classifier.buildClassifier(trainData);
      }

      classifyUnseenData(classifier, unseenData);
    } else {
      classifier = loadModel(params.get("model") != null ? params.get("model").get(0) : "id3");
      testModel();
    }

    if (isSaveModel) {
      saveModel();
    }
  }
}