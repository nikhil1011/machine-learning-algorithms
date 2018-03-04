import java.util.*;

public class LogisticRegressionClassifier {
	List<Double> weights;
	
	public LogisticRegressionClassifier(int n) throws Exception {
		if(n<1) {
			throw new Exception("The number of weights should be at least 1");
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
			for(int i = 0; i<weights.size(); i++) {
				double delta = 0;
				double penalty = 0;
				double updatedWeight = 0;

				for(List<Double> dataPoint: dataSet.keySet()) {
					Integer targetClass = dataSet.get(dataPoint);
					double probabilityOfDataPoint = classifyDataPoint(dataPoint);
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
	
	public double sigmoidFunction(double num) {
		double sigmoid = (Math.exp(num)/ (1 + Math.exp(num)));
		
		return sigmoid;
	}
	
	public static void main(String[] args) {
		
		Map<List<Double>, Integer> dataSet = new HashMap<>();
		
		List<Double> point1 = new ArrayList<>();
		point1.add(1.0);
		point1.add(1.0);
		
		List<Double> point2 = new ArrayList<>();
		point2.add(-2.0);
		point2.add(1.0);
		
		List<Double> point3 = new ArrayList<>();
		point3.add(1.5);
		point3.add(-0.5);
		
		List<Double> point4 = new ArrayList<>();
		point4.add(-2.0);
		point4.add(-1.0);
		
		List<Double> point5 = new ArrayList<>();
		point5.add(-1.0);
		point5.add(-1.5);
		
		List<Double> point6 = new ArrayList<>();
		point6.add(2.0);
		point6.add(-2.0);
		
		dataSet.put(point1, 1);
		dataSet.put(point2, 1);
		dataSet.put(point3, 1);
		dataSet.put(point4, -1);
		dataSet.put(point5, -1);
//		dataSet.put(point6, -1);
		
		double lambda = 0.1;
		int iterations = 1000;
		
		try {
			LogisticRegressionClassifier lrClassifier = new LogisticRegressionClassifier(2);
			lrClassifier.train(dataSet, iterations, lambda);
			double probability = lrClassifier.classifyDataPoint(point6);
			if(probability>(1-probability)) {
				System.out.println("One");
			}
			else {
				System.out.println("Zero");
			}
		} 
		catch (Exception e) {
			System.out.println(e.getMessage());
		}

	}
	
}
