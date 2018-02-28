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
	
	public List<Double> train(List<List<Double>> inputs){
		if(inputs == null) {
			return null;
		}
		if(inputs.get(0).size()!=weights.size() - 1) {
			return null;
		}
		
		return weights;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	}

}
