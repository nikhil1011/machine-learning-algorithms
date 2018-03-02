import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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
		
//		Perceptron perceptron = new Perceptron(2);
//		try {
//			perceptron.train(dataSet);
//		} catch (Exception e) {
//			System.out.println(e.getMessage());
//		}
		//sample run 1 values 0.22310717402523456 0.4235661829513714 0.9766066668304185
		//sample run 2 values 0.31335556915409285 0.26708008127891536 0.6193093264629309
//		perceptron.printWeights();
		dataSet.clear();
		
		String folderPath = "src/hw2_train/train/myham";
		Map<Set<String>, List<Map<String, Integer>>> hamVocabularyAndEmailsWordCounts = extractVocabularyAndEmailsWordCounts(folderPath);
		Map.Entry<Set<String>, List<Map<String,Integer>>> hamMapEntry = hamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		Set<String> vocabulary = new HashSet<>();
		vocabulary.addAll(hamMapEntry.getKey());
		List<Map<String, Integer>> hamMapEntryEmailsWordsCounts = new ArrayList<>();
		hamMapEntryEmailsWordsCounts.addAll(hamMapEntry.getValue());
		hamMapEntry = hamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		
		folderPath = "src/hw2_train/train/myspam";
		Map<Set<String>, List<Map<String, Integer>>> spamVocabularyAndEmailsWordCounts = extractVocabularyAndEmailsWordCounts(folderPath);
		Map.Entry<Set<String>, List<Map<String,Integer>>> spamMapEntry = spamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		vocabulary.addAll(spamMapEntry.getKey());
		List<Map<String, Integer>> spamMapEntryEmailsWordsCounts = new ArrayList<>();
		spamMapEntryEmailsWordsCounts.addAll(spamMapEntry.getValue());
		spamMapEntry = spamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		
		hamVocabularyAndEmailsWordCounts = new HashMap<>();
		hamVocabularyAndEmailsWordCounts.put(vocabulary, hamMapEntryEmailsWordsCounts);
		
		spamVocabularyAndEmailsWordCounts = new HashMap<>();
		spamVocabularyAndEmailsWordCounts.put(vocabulary, spamMapEntryEmailsWordsCounts);
		
		constructFeatureVectorMaps(hamVocabularyAndEmailsWordCounts);
		List<Map<String, Integer>> hamEmailFeatureMaps = hamMapEntry.getValue();
		addToDataSet(dataSet, hamEmailFeatureMaps, 1);
		
		constructFeatureVectorMaps(spamVocabularyAndEmailsWordCounts);
		List<Map<String, Integer>> spamEmailFeatureMaps = spamVocabularyAndEmailsWordCounts.entrySet().iterator().next().getValue();
		addToDataSet(dataSet, spamEmailFeatureMaps, -1);
		
		Map<String, Integer> hamFirstEmailMap = hamEmailFeatureMaps.get(0);
		int hamSize = hamFirstEmailMap.size();
//		Set<String> firstEmailMapSet = new TreeSet<String>();
//		firstEmailMapSet.addAll(firstEmailMap.keySet());
		Map<String, Integer> spamFirstEmailMap = spamEmailFeatureMaps.get(0);
		int spamSize = hamFirstEmailMap.size();
		if(hamSize!=spamSize) {
			System.out.println("WOOOAAAAH!!!");
		}
		
		Perceptron perceptron = new Perceptron(hamSize);
		try {
			perceptron.train(dataSet);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println(e.getMessage());
		}
		perceptron.printWeights();
		return;
	}

	private static void addToDataSet(Map<List<Double>, Integer> dataSet,
			List<Map<String, Integer>> hamEmailFeatureMaps, int cls) {
		List<Double> dataPoint;
		for(Map<String, Integer> featureMap: hamEmailFeatureMaps) {
			if(!(featureMap instanceof TreeMap)) {
				System.out.println("Woah!");
			}
//			if(featureMap.size()!=size) {
//				System.out.println("WOOOAH!!");
//			}
			dataPoint = new ArrayList<>();
			for(String word: featureMap.keySet()) {
				dataPoint.add((double)featureMap.get(word));
			}
			dataSet.put(dataPoint, cls);
		}
	}
	
	public static void constructFeatureVectorMaps(Map<Set<String>, List<Map<String, Integer>>> vocabularyAndEmailsWordCounts) {
		Map.Entry<Set<String>, List<Map<String, Integer>>> firstEntry = vocabularyAndEmailsWordCounts.entrySet().iterator().next();
		Set<String> vocabulary = firstEntry.getKey();
		List<Map<String, Integer>> emailWordsCounts = firstEntry.getValue();
		
		for(Map<String, Integer> wordCount: emailWordsCounts) {
			for(String word: vocabulary) {
				if(!wordCount.containsKey(word)) {
					wordCount.put(word, 0);
				}
			}
		}
		
	}
	
	public static Map<Set<String>, List<Map<String, Integer>>> extractVocabularyAndEmailsWordCounts(String folderPath) {
		File folder = new File(folderPath);
		File[] listOfFiles = folder.listFiles();
		
		Set<String> vocabulary = new HashSet<>();
		List<Map<String,Integer>> wordCountsOfAllEmails = new ArrayList<>();
		
		for(File file: listOfFiles) {
			String fileName = folder + "\\" + file.getName();
			try {
				String text = readFile(fileName, StandardCharsets.ISO_8859_1);
				Map<String,Integer> wordsOfThisEmail = extractWords(text);
				vocabulary.addAll(wordsOfThisEmail.keySet());
				wordCountsOfAllEmails.add(wordsOfThisEmail);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.out.println(e.getMessage());
			}
		}
		
		Map<Set<String>, List<Map<String, Integer>>> result = new HashMap<>();
		result.put(vocabulary, wordCountsOfAllEmails);
		
		return result;
	}
	
	public static String readFile(String path, Charset encoding) throws IOException 
	{
	  byte[] encoded = Files.readAllBytes(Paths.get(path));
	  return new String(encoded, encoding);
	}
	
	public static Map<String, Integer> extractWords(String input) {
		Pattern p = Pattern.compile("[\\w']+");
		Matcher m = p.matcher(input);
		Map<String,Integer> wordCount = new TreeMap<>();
		
		while(m.find()) {
			String currentWord = input.substring(m.start(), m.end());
			if(wordCount.containsKey(currentWord)) {
				wordCount.put(currentWord, wordCount.get(currentWord) + 1);
			}
			else {
				wordCount.put(currentWord, 1);
			}
		}
		
		return wordCount;
	}
	
}
