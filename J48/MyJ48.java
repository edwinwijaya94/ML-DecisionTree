package J48;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import J48.Classifier.*;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

public class MyJ48 extends Classifier implements OptionHandler, Drawable,
  Matchable, WeightedInstancesHandler,
  AdditionalMeasureProducer{

  /** The decision tree */
  protected ClassifierTree m_root;

  /** Unpruned tree? */
  protected boolean m_unpruned = false;

  /** Collapse tree? */
  protected boolean m_collapseTree = true;

  /** Confidence level */
  protected float m_CF = 0.25f;

  /** Minimum number of instances */
  protected int m_minNumObj = 2;

  /** Use MDL correction? */
  protected boolean m_useMDLcorrection = true;

  /**
   * Determines whether probabilities are smoothed using Laplace correction when
   * predictions are generated
   */
  protected boolean m_useLaplace = false;

  /** Use reduced error pruning? */
  protected boolean m_reducedErrorPruning = false;

  /** Number of folds for reduced error pruning. */
  protected int m_numFolds = 3;

  /** Binary splits on nominal attributes? */
  protected boolean m_binarySplits = false;

  /** Subtree raising to be performed? */
  protected boolean m_subtreeRaising = true;

  /** Cleanup after the tree has been built. */
  protected boolean m_noCleanup = false;

  /** Random number seed for reduced-error pruning. */
  protected int m_Seed = 1;

  /** Do not relocate split point to actual data value */
  protected boolean m_doNotMakeSplitPointActualValue;

  /**
   * Returns default capabilities of the classifier.
   * 
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result;

    result = new Capabilities(this);
    result.disableAll();
    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    // instances
    result.setMinimumNumberInstances(0);

    return result;
  }

  /**
   * Generates the classifier.
   * 
   * @param instances the data to train the classifier with
   * @throws Exception if classifier can't be built successfully
   */
  @Override
  public void buildClassifier(Instances instances) throws Exception {

    ModelSelection modSelection;
    
    modSelection = new C45ModelSelection(m_minNumObj, instances);
    
    m_root = new C45PruneableClassifierTree(modSelection, !m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup);
    
    m_root.buildClassifier(instances);
    if (m_binarySplits) {
      ((BinC45ModelSelection) modSelection).cleanup();
    } else {
      ((C45ModelSelection) modSelection).cleanup();
    }
  }

  /**
   * Classifies an instance.
   * 
   * @param instance the instance to classify
   * @return the classification for the instance
   * @throws Exception if instance can't be classified successfully
   */
  @Override
  public double classifyInstance(Instance instance) throws Exception {

    return m_root.classifyInstance(instance);
  }

  /**
   * Returns class probabilities for an instance.
   * 
   * @param instance the instance to calculate the class probabilities for
   * @return the class probabilities
   * @throws Exception if distribution can't be computed successfully
   */
  @Override
  public final double[] distributionForInstance(Instance instance)
    throws Exception {

    return m_root.distributionForInstance(instance, m_useLaplace);
  }

  /**
   * Returns the type of graph this classifier represents.
   * 
   * @return Drawable.TREE
   */
  @Override
  public int graphType() {
    return Drawable.TREE;
  }

  /**
   * Returns graph describing the tree.
   * 
   * @return the graph describing the tree
   * @throws Exception if graph can't be computed
   */
  @Override
  public String graph() throws Exception {

    return m_root.graph();
  }

  /**
   * Returns tree in prefix order.
   * 
   * @return the tree in prefix order
   * @throws Exception if something goes wrong
   */
  @Override
  public String prefix() throws Exception {

    return m_root.prefix();
  }

  /**
   * Get the value of Seed.
   * 
   * @return Value of Seed.
   */
  public int getSeed() {

    return m_Seed;
  }

  /**
   * Set the value of Seed.
   * 
   * @param newSeed Value to assign to Seed.
   */
  public void setSeed(int newSeed) {

    m_Seed = newSeed;
  }

  /**
   * Get the value of useLaplace.
   * 
   * @return Value of useLaplace.
   */
  public boolean getUseLaplace() {

    return m_useLaplace;
  }

  /**
   * Set the value of useLaplace.
   * 
   * @param newuseLaplace Value to assign to useLaplace.
   */
  public void setUseLaplace(boolean newuseLaplace) {

    m_useLaplace = newuseLaplace;
  }

  /**
   * Get the value of useMDLcorrection.
   * 
   * @return Value of useMDLcorrection.
   */
  public boolean getUseMDLcorrection() {

    return m_useMDLcorrection;
  }

  /**
   * Set the value of useMDLcorrection.
   * 
   * @param newuseMDLcorrection Value to assign to useMDLcorrection.
   */
  public void setUseMDLcorrection(boolean newuseMDLcorrection) {

    m_useMDLcorrection = newuseMDLcorrection;
  }

  /**
   * Returns the size of the tree
   * 
   * @return the size of the tree
   */
  public double measureTreeSize() {
    return m_root.numNodes();
  }

  /**
   * Returns the number of leaves
   * 
   * @return the number of leaves
   */
  public double measureNumLeaves() {
    return m_root.numLeaves();
  }

  /**
   * Returns the number of rules (same as number of leaves)
   * 
   * @return the number of rules
   */
  public double measureNumRules() {
    return m_root.numLeaves();
  }

  /**
   * Returns an enumeration of the additional measure names
   * 
   * @return an enumeration of the measure names
   */
  @Override
  public Enumeration<String> enumerateMeasures() {
    Vector<String> newVector = new Vector<String>(3);
    newVector.addElement("measureTreeSize");
    newVector.addElement("measureNumLeaves");
    newVector.addElement("measureNumRules");
    return newVector.elements();
  }

  /**
   * Returns the value of the named measure
   * 
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String additionalMeasureName) {
    if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
      return measureNumRules();
    } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
      return measureTreeSize();
    } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
      return measureNumLeaves();
    } else {
      throw new IllegalArgumentException(additionalMeasureName
        + " not supported (j48)");
    }
  }

  /**
   * Get the value of unpruned.
   * 
   * @return Value of unpruned.
   */
  public boolean getUnpruned() {

    return m_unpruned;
  }

  /**
   * Set the value of unpruned. Turns reduced-error pruning off if set.
   * 
   * @param v Value to assign to unpruned.
   */
  public void setUnpruned(boolean v) {

    if (v) {
      m_reducedErrorPruning = false;
    }
    m_unpruned = v;
  }
  
  /**
   * Get the value of collapseTree.
   * 
   * @return Value of collapseTree.
   */
  public boolean getCollapseTree() {

    return m_collapseTree;
  }

  /**
   * Set the value of collapseTree.
   * 
   * @param v Value to assign to collapseTree.
   */
  public void setCollapseTree(boolean v) {

    m_collapseTree = v;
  }

  /**
   * Get the value of CF.
   * 
   * @return Value of CF.
   */
  public float getConfidenceFactor() {

    return m_CF;
  }

  /**
   * Set the value of CF.
   * 
   * @param v Value to assign to CF.
   */
  public void setConfidenceFactor(float v) {

    m_CF = v;
  }

  /**
   * Get the value of minNumObj.
   * 
   * @return Value of minNumObj.
   */
  public int getMinNumObj() {

    return m_minNumObj;
  }

  /**
   * Set the value of minNumObj.
   * 
   * @param v Value to assign to minNumObj.
   */
  public void setMinNumObj(int v) {

    m_minNumObj = v;
  }

  /**
   * Get the value of reducedErrorPruning.
   * 
   * @return Value of reducedErrorPruning.
   */
  public boolean getReducedErrorPruning() {

    return m_reducedErrorPruning;
  }

  /**
   * Set the value of reducedErrorPruning. Turns unpruned trees off if set.
   * 
   * @param v Value to assign to reducedErrorPruning.
   */
  public void setReducedErrorPruning(boolean v) {

    if (v) {
      m_unpruned = false;
    }
    m_reducedErrorPruning = v;
  }

  /**
   * Get the value of numFolds.
   * 
   * @return Value of numFolds.
   */
  public int getNumFolds() {

    return m_numFolds;
  }

  /**
   * Set the value of numFolds.
   * 
   * @param v Value to assign to numFolds.
   */
  public void setNumFolds(int v) {

    m_numFolds = v;
  }

  /**
   * Get the value of binarySplits.
   * 
   * @return Value of binarySplits.
   */
  public boolean getBinarySplits() {

    return m_binarySplits;
  }

  /**
   * Set the value of binarySplits.
   * 
   * @param v Value to assign to binarySplits.
   */
  public void setBinarySplits(boolean v) {

    m_binarySplits = v;
  }

  /**
   * Get the value of subtreeRaising.
   * 
   * @return Value of subtreeRaising.
   */
  public boolean getSubtreeRaising() {

    return m_subtreeRaising;
  }

  /**
   * Set the value of subtreeRaising.
   * 
   * @param v Value to assign to subtreeRaising.
   */
  public void setSubtreeRaising(boolean v) {

    m_subtreeRaising = v;
  }

  /**
   * Check whether instance data is to be saved.
   * 
   * @return true if instance data is saved
   */
  public boolean getSaveInstanceData() {

    return m_noCleanup;
  }

  /**
   * Set whether instance data is to be saved.
   * 
   * @param v true if instance data is to be saved
   */
  public void setSaveInstanceData(boolean v) {

    m_noCleanup = v;
  }

  /**
   * Gets the value of doNotMakeSplitPointActualValue.
   * 
   * @return the value
   */
  public boolean getDoNotMakeSplitPointActualValue() {
    return m_doNotMakeSplitPointActualValue;
  }

  /**
   * Sets the value of doNotMakeSplitPointActualValue.
   * 
   * @param m_doNotMakeSplitPointActualValue the value to set
   */
  public void setDoNotMakeSplitPointActualValue(
    boolean m_doNotMakeSplitPointActualValue) {
    this.m_doNotMakeSplitPointActualValue = m_doNotMakeSplitPointActualValue;
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

  /**
   * Builds the classifier to generate a partition.
   */
  public void generatePartition(Instances data) throws Exception {

    buildClassifier(data);
  }

  /**
   * Computes an array that indicates node membership.
   */
//  public double[] getMembershipValues(Instance inst) throws Exception {
//
//    return m_root.getMembershipValues(inst);
//  }

  /**
   * Returns the number of elements in the partition.
   */
  public int numElements() throws Exception {

    return m_root.numNodes();
  }

  /**
   * Main method for testing this class
   * 
   * @param argv the commandline options
   */
  public static void main(String[] argv) {
    runClassifier(new MyJ48(), argv);
  }
}