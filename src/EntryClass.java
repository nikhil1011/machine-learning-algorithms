import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class EntryClass {

	public static void main(String[] args) {
		
		if(args.length!=4) {
			System.out.println("Four arguments are needed exactly");
			return;
		}
		
		System.out.println("------This program will print reported accuracies on both Perceptron and Logistic Regression.------");
		
		int choice = Integer.parseInt(args[0]);
		String showValidation = args[1];
		int noOfIterations = Integer.parseInt(args[2]);
		double lambda = Double.parseDouble(args[3]);
		
		String hamTrainingFolder = "";
		String hamValidationFolder = "";
		String hamTestFolder = "";
		
		String spamTrainingFolder = "";
		String spamValidationFolder = "";
		String spamTestFolder = "";
		
		if(choice == 1) {
			
			hamTrainingFolder = "src/hw2_train/train/ham/training_set";
			hamValidationFolder = "src/hw2_train/train/ham/validation_set";
			hamTestFolder = "src/hw2_test/test/ham/";
			
			spamTrainingFolder = "src/hw2_train/train/spam/training_set";
			spamValidationFolder = "src/hw2_train/train/spam/validation_set";
			spamTestFolder = "src/hw2_test/test/spam";
			
		}
		else if(choice ==2) {
			hamTrainingFolder = "src/enron1_train/enron1/train/ham/training_set";
			hamValidationFolder = "src/enron1_train/enron1/train/ham/validation_set";
			hamTestFolder = "src/enron1_test/enron1/test/ham";
			
			spamTrainingFolder = "src/enron1_train/enron1/train/spam/training_set";;
			spamValidationFolder = "src/enron1_train/enron1/train/spam/validation_set";;
			spamTestFolder = "src/enron1_test/enron1/test/spam";
		}
		else if(choice == 3) {
			hamTrainingFolder = "src/enron4_train/enron4/train/ham/training_set";
			hamValidationFolder = "src/enron4_train/enron4/train/ham/validation_set";
			hamTestFolder = "src/enron4_test/enron4/test/ham";
			
			spamTrainingFolder = "src/enron4_train/enron4/train/spam/training_set";;
			spamValidationFolder = "src/enron4_train/enron4/train/spam/validation_set";;
			spamTestFolder = "src/enron4_test/enron4/test/spam";
		}
		
		else {
			System.out.println("Wrong choice. Restart the program.");
			return;
		}
		
		System.out.println("------Reading and constructing training and validation feature vectors.------");
		
		Map<List<Double>, Integer> dataSet = new HashMap<>();
		Set<String> vocabulary = new HashSet<>();
		String folderPath = hamTrainingFolder; //"src/hw2_train/train/ham/training_set";
		Map<Set<String>, List<Map<String, Integer>>> hamVocabularyAndEmailsWordCounts = Perceptron.extractVocabularyAndEmailsWordCounts(folderPath);
		Map.Entry<Set<String>, List<Map<String,Integer>>> hamMapEntry = hamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		vocabulary.addAll(hamMapEntry.getKey());
		List<Map<String, Integer>> hamMapEntryEmailsWordsCounts = new ArrayList<>();
		hamMapEntryEmailsWordsCounts.addAll(hamMapEntry.getValue());
		hamMapEntry = hamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		
		folderPath = spamTrainingFolder; //"src/hw2_train/train/spam/training_set";
		Map<Set<String>, List<Map<String, Integer>>> spamVocabularyAndEmailsWordCounts = Perceptron.extractVocabularyAndEmailsWordCounts(folderPath);
		Map.Entry<Set<String>, List<Map<String,Integer>>> spamMapEntry = spamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		vocabulary.addAll(spamMapEntry.getKey());
		List<Map<String, Integer>> spamMapEntryEmailsWordsCounts = new ArrayList<>();
		spamMapEntryEmailsWordsCounts.addAll(spamMapEntry.getValue());
		spamMapEntry = spamVocabularyAndEmailsWordCounts.entrySet().iterator().next();
		
		hamVocabularyAndEmailsWordCounts = new HashMap<>();
		hamVocabularyAndEmailsWordCounts.put(vocabulary, hamMapEntryEmailsWordsCounts);
		
		spamVocabularyAndEmailsWordCounts = new HashMap<>();
		spamVocabularyAndEmailsWordCounts.put(vocabulary, spamMapEntryEmailsWordsCounts);
		
		Perceptron.constructFeatureVectorMaps(hamVocabularyAndEmailsWordCounts);
		List<Map<String, Integer>> hamEmailFeatureMaps = hamMapEntry.getValue();
		Perceptron.addToDataSet(dataSet, hamEmailFeatureMaps, 1);
		
		Perceptron.constructFeatureVectorMaps(spamVocabularyAndEmailsWordCounts);
		List<Map<String, Integer>> spamEmailFeatureMaps = spamVocabularyAndEmailsWordCounts.entrySet().iterator().next().getValue();
		Perceptron.addToDataSet(dataSet, spamEmailFeatureMaps, -1);
		
		Map<String, Integer> hamFirstEmailMap = hamEmailFeatureMaps.get(0);
		int hamSize = hamFirstEmailMap.size();
		Map<String, Integer> spamFirstEmailMap = spamEmailFeatureMaps.get(0);
		int spamSize = spamFirstEmailMap.size();
		
		folderPath = hamValidationFolder; //"src/hw2_train/train/ham/validation_set";
		List<Map<String, Integer>> validationHamEmailFeatureMaps = Perceptron.getEmailFeatureMaps(folderPath, vocabulary);
		folderPath = spamValidationFolder; //"src/hw2_train/train/spam/validation_set";
		List<Map<String, Integer>> validationSpamEmailFeatureMaps = Perceptron.getEmailFeatureMaps(folderPath, vocabulary);
		
		System.out.println("No of iterations is equal to:" + noOfIterations);
		
//		System.out.println("-------Begin Perceptron-------");
//		
//		Perceptron perceptron;
//		if (showValidation.equals("yes")) {
//			System.out.println("**Validation data accuracy for Perceptron**");
//			perceptron = new Perceptron(hamSize);
//			try {
//				perceptron.train(dataSet, noOfIterations);
//			} catch (Exception e) {
//				System.out.println(e.getMessage());
//			}
//			dataSet.clear();
//			Perceptron.addToDataSet(dataSet, validationHamEmailFeatureMaps, 1);
//			Perceptron.addToDataSet(dataSet, validationSpamEmailFeatureMaps, -1);
//			int validationHamTests = validationHamEmailFeatureMaps.size();
//			int validationSpamTests = validationSpamEmailFeatureMaps.size();
//			List<Integer> validationHamTestResults = perceptron.testDataSet(dataSet);
//			int validationHits = validationHamTestResults.get(0);
//			//		int hamTestMisses = testHamTestResults.get(1);
//			double accuracyOfValidationSet = (double) validationHits
//					/ (double) (validationHamTests + validationSpamTests);
//			System.out.println(
//					"Accuracy of Validation Set(on both Ham and Spam combined): " + accuracyOfValidationSet);
//		}
//		
//		dataSet.clear();
//		
//		//training a new perceptron on both training and validation combined
//		Perceptron.addToDataSet(dataSet, hamEmailFeatureMaps, 1);
//		Perceptron.addToDataSet(dataSet, spamEmailFeatureMaps, -1);
//		Perceptron.addToDataSet(dataSet, validationHamEmailFeatureMaps, 1);
//		Perceptron.addToDataSet(dataSet, validationSpamEmailFeatureMaps, -1);
//		
//		Perceptron trainingAndValidationPerceptron = new Perceptron(hamSize);
//		
//		try {
//			trainingAndValidationPerceptron.train(dataSet, noOfIterations);
//		} catch (Exception e) {
//			System.out.println(e.getMessage());
//			e.printStackTrace();
//		}
//		dataSet.clear();
//		
//		//creating test feature maps and testing them on the trained perceptron above
//		System.out.println("Using both training and validation feature vectors to perform perceptron training for test data");
//		folderPath = hamTestFolder;//"src/hw2_test/test/ham/";
//		List<Map<String, Integer>> testHamEmailFeatureMaps = Perceptron.getEmailFeatureMaps(folderPath, vocabulary);
//		folderPath = spamTestFolder; //"src/hw2_test/test/spam";
//		List<Map<String, Integer>> testSpamEmailFeatureMaps = Perceptron.getEmailFeatureMaps(folderPath, vocabulary);
//		
//		int testHamTests = testHamEmailFeatureMaps.size();
//		int testSpamTests = testSpamEmailFeatureMaps.size();
//		
//		Perceptron.addToDataSet(dataSet, testHamEmailFeatureMaps, 1);
//		
//		List<Integer> testHamTestResults = trainingAndValidationPerceptron.testDataSet(dataSet);
//		int hamTestHits = testHamTestResults.get(0);
//		
//		dataSet.clear();
//		Perceptron.addToDataSet(dataSet, testSpamEmailFeatureMaps, -1);
//		List<Integer> testSpamTestResults = trainingAndValidationPerceptron.testDataSet(dataSet);
//		
//		int spamTestHits = testSpamTestResults.get(0);
//		
//		double accuracyOfTestSet = (double)hamTestHits / (double)(testHamTests);
//		System.out.println("Accuracy of Test Ham Set: " + accuracyOfTestSet);
//		
//		accuracyOfTestSet = (double)spamTestHits / (double)(testSpamTests);
//		System.out.println("Accuracy of Test Spam Set: " + accuracyOfTestSet);
//		dataSet.clear();
//		System.out.println("--------End Perceptron----------");
		
		System.out.println("-------Begin Logistic Regression---------");
		LogisticRegressionClassifier lrClassifier = new LogisticRegressionClassifier(hamSize);
		
		Perceptron.addToDataSet(dataSet, hamEmailFeatureMaps, 1);
		Perceptron.addToDataSet(dataSet, spamEmailFeatureMaps, -1);
		
		if(showValidation.equals("yes")) {
			
			try {
				lrClassifier.train(dataSet, noOfIterations, lambda);
			} catch (Exception e) {
				System.out.println(e.getMessage());
				e.printStackTrace();
			}
			
			dataSet.clear();
			
			Perceptron.addToDataSet(dataSet, validationHamEmailFeatureMaps, 1);
			Perceptron.addToDataSet(dataSet, validationSpamEmailFeatureMaps, -1);
			
			List<Integer> validationLRResults = lrClassifier.testDataSet(dataSet);
			int resultsSize = validationLRResults.size();
			
			int positiveHits = validationLRResults.get(resultsSize - 4);
			int positiveMisses = validationLRResults.get(resultsSize - 3);
			int negativeHits = validationLRResults.get(resultsSize - 2);
			int negativeMisses = validationLRResults.get(resultsSize - 1);
			
			System.out.println("done");
		}
	}

}
