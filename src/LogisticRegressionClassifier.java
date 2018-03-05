import java.util.*;

public class LogisticRegressionClassifier {
	List<Double> weights;
	
	public LogisticRegressionClassifier(int n){
		weights = new ArrayList<>();
		
		for(int i = 0; i<=n; i++) {
			weights.add(0.0);
		}
	}
	
	public void train(Map<List<Double>, Integer> dataSet, int iterations, double lambda) throws Exception {
		if(dataSet == null || dataSet.size()<0) {
			throw new Exception("data set should not be empty");
		}
		
		double learningRate = 0.01;
		for(int j = 0; j<iterations; j++) {

//			for(Double feature: dataPoint) {
//				delta += feature*difference;
//			}
//			delta *= learningRate;
			for(int i = 0; i<weights.size(); i++) {
				double delta = 0;
				double penalty = 0;
				double updatedWeight = 0;
				
				for(List<Double> dataPoint: dataSet.keySet()) {
					Integer targetClass = dataSet.get(dataPoint);
					double probabilityOfDataPoint = classifyDataPoint(dataPoint);
					if(probabilityOfDataPoint>=0.5) {
						probabilityOfDataPoint = 1;
					}
					else {
						probabilityOfDataPoint = 0;
					}
					double difference = targetClass - probabilityOfDataPoint;
					if(i == 0) {
						delta += difference;
					}
					else {
						delta += dataPoint.get(i - 1)*difference;
					}
				}
				
				delta *= learningRate;
				penalty = learningRate * lambda * weights.get(i);
				updatedWeight = weights.get(i) + delta - penalty;
				weights.set(i, updatedWeight);
			}
		}
	}
	
	public double classifyDataPoint(List<Double> dataPoint) {
		double sigmoid = weights.get(0);
		
		for(int i = 1; i<weights.size(); i++) {
			sigmoid += weights.get(i)*dataPoint.get(i-1);
		}
		sigmoid = sigmoidFunction(sigmoid);
		return sigmoid;
	}
	
	public List<Integer> testDataSet(Map<List<Double>, Integer> dataSet){
		List<Integer> results = new ArrayList<>();
		
		int positiveHits = 0;
		int negativeHits = 0;
		int positiveMisses = 0;
		int negativeMisses = 0;
		
		for(List<Double> dataPoint: dataSet.keySet()) {
			double outputClassProb = classifyDataPoint(dataPoint);
			int outputClass;
			if(outputClassProb>(1-outputClassProb)) {
				outputClass = 1;
			}
			else {
				outputClass = -1;
			}
			
			int targetClass = dataSet.get(dataPoint);
			if(targetClass == 1) {
				if(outputClass == 1) {
					positiveHits++;
				}
				else {
					positiveMisses++;
				}
			}
			else {
				if(outputClass == -1) {
					negativeHits++;
				}
				else {
					negativeMisses++;
				}
			}
			results.add(outputClass);
		}
		
		results.add(positiveHits);
		results.add(positiveMisses);
		results.add(negativeHits);
		results.add(negativeMisses);
		
		return results;
	}
	
	public double sigmoidFunction(double num) {
		double sigmoid = (Math.exp(num)/ (1 + Math.exp(num)));
		return sigmoid;
	}	
}
