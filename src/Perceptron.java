import java.util.*;
public class Perceptron {
	
	List<Double> weights;
	
	public Perceptron(int n) {
		weights = new ArrayList<>();
		Random random = new Random();
		
		for(int i = 0; i<n; i++) {
			weights.add(random.nextDouble());
		}
	}
	
	public List<Double> train(Map<List<Double>,Integer> inputs){
		if(inputs == null) {
			return null;
		}
		
		Map.Entry<List<Double>, Integer> entry = inputs.entrySet().iterator().next();
		if(entry.getKey().size() != weights.size() - 1) {
			return null;
		}
//		if(inputs.get(0).size()!=weights.size() - 1) {
//			return null;
//		}
		
		boolean misclassified = true;
		while(misclassified) {
			misclassified = false;
			for(List<Double> dataPoint: inputs.keySet()) {
				Integer outputClass = testDataPoint(dataPoint);
			}
		}
		
		return weights;
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
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	}

}
