import java.util.*;

public class LogisticRegressionClassifier {
	List<Double> weights;
	
	public LogisticRegressionClassifier(int n) throws Exception {
		if(n<2) {
			throw new Exception("The number of weights should be at least 2");
		}
		weights = new ArrayList<>();
//		Random random = new Random();
		
		for(int i = 0; i<=n; i++) {
			weights.add(0.0);
		}
	}
	
	public void train(Map<List<Double>, Integer> dataSet, int iterations, double lambda) throws Exception {
		if(dataSet == null || dataSet.size()<0) {
			throw new Exception("data set should not be empty");
		}
		
		double learningRate = 0.1;
		for(int j = 0; j<iterations; j++) {

//			for(Double feature: dataPoint) {
//				delta += feature*difference;
//			}
//			delta *= learningRate;
			for(int i = 0; i<weights.size(); i++) {
				double delta = 0;
				double penalty = 0;
				double updatedWeight = 0;
				
				if(i == 0) {
					for(List<Double> dataPoint: dataSet.keySet()) {
						Integer targetClass = dataSet.get(dataPoint);
						double probabilityOfDataPoint = applySigmoidFunctionForDataPoint(dataPoint);
						double difference = targetClass - probabilityOfDataPoint;
						if(i == 0) {
							delta += difference;
						}
						else {
							
						}
					}
				}
				else {
					for(List<Double> dataPoint: dataSet.keySet()) {
						Integer targetClass = dataSet.get(dataPoint);
						double probabilityOfDataPoint = applySigmoidFunctionForDataPoint(dataPoint);
						double difference = targetClass - probabilityOfDataPoint;
						
					}
				}
				
				delta *= learningRate;
				penalty = learningRate * lambda * weights.get(i);
				updatedWeight = weights.get(i) + delta - penalty;
				weights.set(i, updatedWeight);
			}
		}
	}
	
	public double applySigmoidFunctionForDataPoint(List<Double> dataPoint) {
		double sigmoid = weights.get(0);
		
		for(int i = 1; i<weights.size(); i++) {
			sigmoid += weights.get(i)*dataPoint.get(i-1);
		}
		sigmoid = sigmoidFunction(sigmoid);
		
		return sigmoid;
	}
	
	public double sigmoidFunction(double num) {
		double sigmoid = (Math.exp(num)/ (1 + Math.exp(num)));
		
		return sigmoid;
	}
	
}
