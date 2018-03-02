import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
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
	
	public void train(Map<List<Double>,Integer> inputs, int iterations) throws Exception{
		if(inputs == null || iterations < 0) {
			throw new Exception("Data Set is empty (or) invalid number of iterations");
		}
		
		Map.Entry<List<Double>, Integer> entry = inputs.entrySet().iterator().next();
		if(entry.getKey().size() != weights.size() - 1) {
			throw new Exception("Number of dimensions should be equal to " + (weights.size() - 1));
		}
		
		double learningRate = 0.1;
		while(iterations>0) {
			for(List<Double> currentInput: inputs.keySet()) {
				Integer outputClass = testDataPoint(currentInput);
				Integer targetClass = inputs.get(currentInput);
				
				if(outputClass != targetClass) {
					double constantFactor = learningRate*(targetClass - outputClass);
					double currentWeightChange = constantFactor;
					weights.set(0, weights.get(0) + currentWeightChange);
					
					for(int i = 1; i<weights.size(); i++) {
						currentWeightChange = constantFactor*currentInput.get(i-1);
						weights.set(i, weights.get(i) + currentWeightChange);
					}
				}
				
			}
			iterations--;
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
	
	public List<Integer> testDataSet(Map<List<Double>, Integer> dataSet){
		List<Integer> results = new ArrayList<>();
		int hits = 0;
		int misses = 0;
		
		for(List<Double> dataPoint: dataSet.keySet()) {
			int targetClass = dataSet.get(dataPoint);
			int outputClass = testDataPoint(dataPoint);
			
			if(targetClass != outputClass) {
				misses++;
			}
			else {
				hits++;
			}
		}
		
		results.add(hits);
		results.add(misses);
		return results;
	}
	
	public void printWeights() {
		for(Double weight: weights) {
			System.out.print(weight + " ");
		}
	}
	
	public String getWeightsString() {
		StringBuilder weightsBuilder = new StringBuilder("");
		
		for(Double weight: weights) {
			weightsBuilder.append(weight);
			weightsBuilder.append(" ");
		}
		
		return weightsBuilder.toString();
	}
	
	public static void main(String[] args) throws IOException {
		
		System.out.println("Enter the folder locations for (i)ham training set (ii) ham validation set (iii) ham test set and similarly for spam(in the same order)");
		
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
		List<String> parameters = Arrays.asList(bufferedReader.readLine().split(" "));

		if(parameters.size()!=6) {
			System.out.println("Not the required number of parameters. Exiting.");
			return;
		}
		
		String hamTrainingFolder = parameters.get(0);
		String hamValidationFolder = parameters.get(1);
		String hamTestFolder = parameters.get(2);
		
		String spamTrainingFolder = parameters.get(4);
		String spamValidationFolder = parameters.get(5);
		String spamTestFolder = parameters.get(6);
		
		Map<List<Double>, Integer> dataSet = new HashMap<>();
		Set<String> vocabulary = new HashSet<>();
		
		String folderPath = hamTrainingFolder; //"src/hw2_train/train/ham/training_set";
		Map<Set<String>, List<Map<String, Integer>>> hamVocabularyAndEmailsWordCounts = extractVocabularyAndEmailsWordCounts(folderPath);
		Map.Entry<Set<String>, List<Map<String,Integer>>> hamMapEntry = hamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		vocabulary.addAll(hamMapEntry.getKey());
		List<Map<String, Integer>> hamMapEntryEmailsWordsCounts = new ArrayList<>();
		hamMapEntryEmailsWordsCounts.addAll(hamMapEntry.getValue());
		hamMapEntry = hamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		
		folderPath = spamTrainingFolder; //"src/hw2_train/train/spam/training_set";
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
		Map<String, Integer> spamFirstEmailMap = spamEmailFeatureMaps.get(0);
		int spamSize = spamFirstEmailMap.size();
		
		int noOfIterations = 1000;
		Perceptron perceptron = new Perceptron(hamSize);
		try {
			perceptron.train(dataSet,noOfIterations);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		
//		String resultString = perceptron.getWeightsString();
//		String resultFilePath = "src/hw2_train/train/perceptronweights/all_files_result_attempt_1_02-03-18-02-18.txt/";
//		writeResultToFile(resultString, resultFilePath);
		dataSet.clear();
		
		folderPath = hamValidationFolder; //"src/hw2_train/train/ham/validation_set";
		List<Map<String, Integer>> validationHamEmailFeatureMaps = getEmailFeatureMaps(folderPath, vocabulary);
		folderPath = spamValidationFolder; //"src/hw2_train/train/spam/validation_set";
		List<Map<String, Integer>> validationSpamEmailFeatureMaps = getEmailFeatureMaps(folderPath, vocabulary);
		
		addToDataSet(dataSet, validationHamEmailFeatureMaps, 1);
		addToDataSet(dataSet, validationSpamEmailFeatureMaps, -1);
		
		int validationHamTests = validationHamEmailFeatureMaps.size();
		int validationSpamTests = validationSpamEmailFeatureMaps.size();
		List<Integer> validationHamTestResults = perceptron.testDataSet(dataSet);
		int validationHits = validationHamTestResults.get(0);
//		int hamTestMisses = testHamTestResults.get(1);
		
		dataSet.clear();
		
		folderPath = hamTestFolder;//"src/hw2_test/test/ham/";
		List<Map<String, Integer>> testHamEmailFeatureMaps = getEmailFeatureMaps(folderPath, vocabulary);
		folderPath = spamTestFolder; //"src/hw2_test/test/spam";
		List<Map<String, Integer>> testSpamEmailFeatureMaps = getEmailFeatureMaps(folderPath, vocabulary);
		
		int testHamTests = testHamEmailFeatureMaps.size();
		int testSpamTests = testSpamEmailFeatureMaps.size();
		
		addToDataSet(dataSet, testHamEmailFeatureMaps, 1);
		
		List<Integer> testHamTestResults = perceptron.testDataSet(dataSet);
		int hamTestHits = testHamTestResults.get(0);
		
		dataSet.clear();
		addToDataSet(dataSet, testSpamEmailFeatureMaps, -1);
		List<Integer> testSpamTestResults = perceptron.testDataSet(dataSet);
		
		int spamTestHits = testSpamTestResults.get(0);
		
		System.out.println("No of iterations is equal to:" + noOfIterations);
		
		double accuracyOfValidationSet = (double)validationHits / (double)(validationHamTests + validationSpamTests);
		System.out.println("Accuracy of Validation Set(on both Ham and Spam combined): " + accuracyOfValidationSet);
		
		double accuracyOfTestSet = (double)hamTestHits / (double)(testHamTests);
		System.out.println("Accuracy of Test Ham Set: " + accuracyOfTestSet);
		
		accuracyOfTestSet = (double)spamTestHits / (double)(testSpamTests);
		System.out.println("Accuracy of Test Spam Set: " + accuracyOfTestSet);
		
	}

	private static List<Map<String, Integer>> getEmailFeatureMaps(String folderPath, Set<String> vocabulary) {
		Map<Set<String>, List<Map<String, Integer>>> vocabularyAndEmailsWordCounts = extractVocabularyAndEmailsWordCounts(folderPath);
		Map.Entry<Set<String>, List<Map<String, Integer>>> mapFirstEntry = vocabularyAndEmailsWordCounts.entrySet().iterator().next();
		
		List<Map<String, Integer>> emailsWordCounts = new ArrayList<>();
		emailsWordCounts.addAll(mapFirstEntry.getValue());
		vocabularyAndEmailsWordCounts = new HashMap<>();
		vocabularyAndEmailsWordCounts.put(vocabulary, emailsWordCounts);
		
		constructFeatureVectorMaps(vocabularyAndEmailsWordCounts);
		List<Map<String, Integer>> emailFeatureMaps = vocabularyAndEmailsWordCounts.entrySet().iterator().next().getValue();
		removeWordsNotInVocabulary(emailFeatureMaps, vocabulary);
		
		return emailFeatureMaps;
	}
	
	public static void removeWordsNotInVocabulary(List<Map<String,Integer>> emailWordsCounts, Set<String> vocabulary) {
		//remove words that are not there in vocabulary
		List<String> wordsToBeRemoved = new ArrayList<>();
		for(Map<String, Integer> wordCount: emailWordsCounts) {
			for(String word: wordCount.keySet()) {
				if(!vocabulary.contains(word)) {
					wordsToBeRemoved.add(word);
				}
			}
			
			for(String word: wordsToBeRemoved) {
				wordCount.remove(word);
			}
			
			wordsToBeRemoved.clear();
		}
	}
	private static void writeResultToFile(String resultString, String resultFilePath) {
		try {
			PrintWriter fileWriter = new PrintWriter(resultFilePath, "UTF-8");
			fileWriter.println(resultString);
			fileWriter.flush();
			fileWriter.close();
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}

	private static void addToDataSet(Map<List<Double>, Integer> dataSet,
			List<Map<String, Integer>> hamEmailFeatureMaps, int cls) {
		List<Double> dataPoint;
		for(Map<String, Integer> featureMap: hamEmailFeatureMaps) {
//			if(!(featureMap instanceof TreeMap)) {
//				System.out.println("Woah!");
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
		
		Map<String,Integer> wordCount = Collections.synchronizedMap(new TreeMap<>());
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
