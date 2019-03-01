package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A function that counts the ones in the data
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesEvaluationFunction implements EvaluationFunction {
    /*
    * The number of function evaluations done. Increments every time the value() function is called
    */
	public long fevals = 0;

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        this.fevals = this.fevals + 1;
        Vector data = d.getData();
        double val = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i) == 1) {
                val++;
            }
        }
        return val;
    }
}