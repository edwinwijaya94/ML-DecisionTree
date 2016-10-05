
package J48.Classifier;

import java.io.Serializable;

import weka.core.Instances;
import weka.core.RevisionHandler;

/**
 * Abstract class for model selection criteria.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public abstract class ModelSelection
  implements Serializable, RevisionHandler {

  /** for serialization */
  private static final long serialVersionUID = -4850147125096133642L;

  /**
   * Selects a model for the given dataset.
   *
   * @exception Exception if model can't be selected
   */
  public abstract ClassifierSplitModel selectModel(Instances data) throws Exception;

  /**
   * Selects a model for the given train data using the given test data
   *
   * @exception Exception if model can't be selected
   */
  public ClassifierSplitModel selectModel(Instances train, Instances test) 
       throws Exception {

    throw new Exception("Model selection method not implemented");
  }
}
