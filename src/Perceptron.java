import java.util.*;
public class Perceptron {
	
	List<Double> weights;
	
	public Perceptron(int n) {
		weights = new ArrayList<>();
		Random random = new Random();
		
		for(int i = 0; i<=n; i++) {
			weights.add(random.nextDouble());
		}
	}
	
	public void train(Map<List<Double>,Integer> inputs) throws Exception{
		if(inputs == null) {
			throw new Exception("Data Set is empty");
		}
		
		Map.Entry<List<Double>, Integer> entry = inputs.entrySet().iterator().next();
		if(entry.getKey().size() != weights.size() - 1) {
			throw new Exception("Number of dimensions should be equal to " + (weights.size() - 1));
		}
		
		boolean misclassified = true;
		double learningRate = 0.1;
		while(misclassified) {
			misclassified = false;
			for(List<Double> currentInput: inputs.keySet()) {
				Integer outputClass = testDataPoint(currentInput);
				Integer targetClass = inputs.get(currentInput);
				
				if(outputClass != targetClass) {
					misclassified = true;
					
					double constantFactor = learningRate*(targetClass - outputClass);
					double currentWeightChange = constantFactor;
					weights.set(0, weights.get(0) + currentWeightChange);
					
					for(int i = 1; i<weights.size(); i++) {
						currentWeightChange = constantFactor*currentInput.get(i-1);
						weights.set(i, weights.get(i) + currentWeightChange);
					}
				}
				
			}
		}
	}
	
	public Integer testDataPoint(List<Double> dataPoint) {
		int outputClass;
		
		double result = weights.get(0);
		for(int i = 0; i<dataPoint.size(); i++) {
			result = result + dataPoint.get(i)*weights.get(i+1);
		}
		
		if(result>=0) {
			outputClass = 1;
		}
		else {
			outputClass = -1;
		}
		
		return outputClass;
	}
	
	public void printWeights() {
		for(Double weight: weights) {
			System.out.print(weight + " ");
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
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
		dataSet.put(point6, -1);
		
		Perceptron perceptron = new Perceptron(2);
		try {
			perceptron.train(dataSet);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println(e.getMessage());
		}
		//sample run 1 values 0.22310717402523456 0.4235661829513714 0.9766066668304185
		//sample run 2 values 0.31335556915409285 0.26708008127891536 0.6193093264629309
		perceptron.printWeights();
		return;
	}

}
