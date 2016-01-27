import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

public class Main {
	public static final int CROSS_VALID_K = 8;
	public static final int THREADS_NUM = 4;
	public static final int N_INPUT = 2;
	public static final String[] CAT_LABELS = {"A", "B", "C"};
	public static final int CAT_NUM = CAT_LABELS.length;

	public static void main(String[] args) {
		List<double[]> dataTrain = readData("data/2d.trn.dat");
		List<double[]> dataTest = readData("data/2d.tst.dat");

//		test1(dataTrain);
//		test2(dataTrain);
//		test3(dataTrain);
//		test4(dataTrain);
//		test5(dataTrain);
		testBest(dataTrain, dataTest);
	}
	
	public static void test1(List<double[]> dataTrain){
		/* Test No.1: Optimal network architecture - activation function correctness:
		 * -> 1-2 layers
		 * -> 2/5/10/20/50/100 neurons on hidden layers
		 * -> all activation functions combinations
		 * -> 100 epoch limit
		 */
		
		int maxLayers = 4;
		int[] nNums = {2, 5, 10, 20, 50};
		int fCount = 4;		// Number of tested activation functions
		System.out.println("Number of layers, number of neurons in each hidden layer, activation function in each layer, avg.est.err, avg.valid.err");
		for(int numLayers = 1; numLayers < maxLayers+1; numLayers++){
			for(int n = 0; n < Math.pow(nNums.length, numLayers-1); n++){
				for(int f = 0; f < Math.pow(fCount, numLayers); f++){
					// Generate numbers of hidden neurons
					int[] numHidNeurons = new int[maxLayers-1];
					for(int i = 0; i < maxLayers-1; i++){
						numHidNeurons[i] = nNums[
											( n / (int)Math.round(Math.pow(nNums.length, i)) )
											% nNums.length ];
					}
					// Generate activation functions
					int[] activFunctions = new int[maxLayers];
					for(int i = 0; i < maxLayers; i++){
						activFunctions[i] = ( f / (int)Math.round(Math.pow(fCount, i)) )
											% fCount;
					}
					// Run test
					MLPParameters params = new MLPParameters();
					params.numLayers = numLayers;
					params.numHidNeurons = numHidNeurons;
					params.activFunctions = activFunctions;
					CrossValidation cv = new CrossValidation(CROSS_VALID_K, params);
					cv.run(dataTrain);
					// Print results
					System.out.print(numLayers+"\t");
					for(int i = maxLayers-2; i >= 0; i--){ System.out.print(numHidNeurons[i]+"\t"); }
					for(int i = maxLayers-1; i >= 0; i--){
						String fName = "";
						switch (activFunctions[i]) {
							case MLPParameters.F_SIGMOID: fName = "sigmoid"; break;
							case MLPParameters.F_TANH: fName = "tanh"; break;
							case MLPParameters.F_LINEAR: fName = "linear"; break;
							case MLPParameters.F_SMOOTH_R_LIN: fName = "smooth"; break;
						}
						System.out.print(fName+"\t");
					}
					System.out.println(cv.getAvgEE()+"\t"+cv.getAvgVE());
				}
			}
		}
	}
	
	public static void test2(List<double[]> dataTrain){
		/* Test No.2: Optimal network architecture - number of layers and neurons:
		 * -> 1-4 layers
		 * -> 2/5/10/20/50 neurons on hidden layers
		 * -> all sigmoid and tanh combinations
		 * -> 100 epoch limit
		 */
		
		int maxLayers = 4;
		int[] nNums = {2, 5, 10, 20, 50};
		int fCount = 2;		// Number of tested activation functions
		System.out.println("Number of layers, number of neurons in each hidden layer, activation function in each layer, avg.est.err, avg.valid.err");
		for(int numLayers = 1; numLayers < maxLayers+1; numLayers++){
			for(int n = 0; n < Math.pow(nNums.length, numLayers-1); n++){
				for(int f = 0; f < Math.pow(fCount, numLayers); f++){
					// Generate numbers of hidden neurons
					int[] numHidNeurons = new int[maxLayers-1];
					for(int i = 0; i < maxLayers-1; i++){
						numHidNeurons[i] = nNums[
											( n / (int)Math.round(Math.pow(nNums.length, i)) )
											% nNums.length ];
					}
					// Generate activation functions
					int[] activFunctions = new int[maxLayers];
					for(int i = 0; i < maxLayers; i++){
						activFunctions[i] = ( f / (int)Math.round(Math.pow(fCount, i)) )
											% fCount;
					}
					// Run test
					MLPParameters params = new MLPParameters();
					params.numLayers = numLayers;
					params.numHidNeurons = numHidNeurons;
					params.activFunctions = activFunctions;
					CrossValidation cv = new CrossValidation(CROSS_VALID_K, params);
					cv.run(dataTrain);
					// Print results
					System.out.print(numLayers+"\t");
					for(int i = maxLayers-2; i >= 0; i--){ System.out.print(numHidNeurons[i]+"\t"); }
					for(int i = maxLayers-1; i >= 0; i--){
						String fName = "";
						switch (activFunctions[i]) {
							case MLPParameters.F_SIGMOID: fName = "sigmoid"; break;
							case MLPParameters.F_TANH: fName = "tanh"; break;
							case MLPParameters.F_LINEAR: fName = "linear"; break;
							case MLPParameters.F_SMOOTH_R_LIN: fName = "smooth"; break;
						}
						System.out.print(fName+"\t");
					}
					System.out.println(cv.getAvgEE()+"\t"+cv.getAvgVE());
				}
			}
		}
	}

	public static void test3(List<double[]> dataTrain){
		/* Test No.3: Optimal network architecture - fine-tuning:
		 * -> 2-3 layers
		 * -> 10/20/35/50 neurons on hidden layers
		 * -> only selected activation functions combinations:
		 *    -> sig-sig, tanh-sig for 2-layer networks
		 *    -> sig-sig-sig, sig-tanh-sig for 3-layer networks
		 * -> 250 epoch limit
		 */

		int minLayers = 2, maxLayers = 3;
		int[] nNums = {10, 20, 35, 50};
		System.out.println("Number of layers, number of neurons in each hidden layer, activation function in each layer, avg.est.err, avg.valid.err");
		for(int numLayers = minLayers; numLayers < maxLayers+1; numLayers++){
			for(int n = 0; n < Math.pow(nNums.length, numLayers-1); n++){
				for(int f = 0; f < 2; f++){
					// Generate numbers of hidden neurons
					int[] numHidNeurons = new int[maxLayers-1];
					for(int i = 0; i < maxLayers-1; i++){
						numHidNeurons[i] = nNums[
											( n / (int)Math.round(Math.pow(nNums.length, i)) )
											% nNums.length ];
					}
					// Generate activation functions (only specified selection)
					int[] activFunctions = {};
					if(numLayers == 2 && f == 0){ activFunctions = new int[]{MLPParameters.F_SIGMOID, MLPParameters.F_SIGMOID, 0}; }
					if(numLayers == 2 && f == 1){ activFunctions = new int[]{MLPParameters.F_TANH, MLPParameters.F_SIGMOID, 0}; }
					if(numLayers == 3 && f == 0){ activFunctions = new int[]{MLPParameters.F_SIGMOID, MLPParameters.F_SIGMOID, MLPParameters.F_SIGMOID}; }
					if(numLayers == 3 && f == 1){ activFunctions = new int[]{MLPParameters.F_SIGMOID, MLPParameters.F_TANH, MLPParameters.F_SIGMOID}; }
					// Run test
					MLPParameters params = new MLPParameters();
					params.numLayers = numLayers;
					params.numHidNeurons = numHidNeurons;
					params.activFunctions = activFunctions;
					CrossValidation cv = new CrossValidation(CROSS_VALID_K, params);
					cv.run(dataTrain);
					// Print results
					System.out.print(numLayers+"\t");
					for(int i = maxLayers-2; i >= 0; i--){ System.out.print(numHidNeurons[i]+"\t"); }
					for(int i = maxLayers-1; i >= 0; i--){
						String fName = "";
						switch (activFunctions[i]) {
							case MLPParameters.F_SIGMOID: fName = "sigmoid"; break;
							case MLPParameters.F_TANH: fName = "tanh"; break;
						}
						System.out.print(fName+"\t");
					}
					System.out.println(cv.getAvgEE()+"\t"+cv.getAvgVE());
				}
			}
		}
	}

	public static void test4(List<double[]> dataTrain){
		/* Test No.4: Optimal network parameters:
		 * -> one chosen architecture: 1 hidden layer of 35 neurons, sig-sig activation functions
		 * -> alpha: 0.005-0.2
		 * -> mi: 0-0.6
		 * -> epsilon: 0-0.2
		 * -> dEpsilon: 1-0.91
		 * -> 250 epoch limit
		 */

		int count = 0;
		double[] alphaOpts = {0.005, 0.01, 0.05, 0.1, 0.2};
		double[] miOpts = {0, 0.01, 0.1, 0.2, 0.4, 0.6};
		double[] epsilonOpts = {0, 0.0001, 0.001, 0.01, 0.1, 0.2};
		double maxDEpsilon = 1, minDEpsilon = 0.9;
		System.out.println("alpha, mi, epsilon, dEpsilon, avg.est.err, avg.valid.err");
		for(int a = 0; a < alphaOpts.length; a++){
			for(int m = 0; m < miOpts.length; m++){
				for(int e = 0; e < epsilonOpts.length; e++){
					for(double de = maxDEpsilon; de > minDEpsilon; de -= 0.01){
						if(++count < 159){ continue; }	//#DEBUG
						// Set parameters
						MLPParameters params = new MLPParameters();
						params.alpha = alphaOpts[a];		params.mi = miOpts[m];
						params.epsilon = epsilonOpts[e];	params.dEpsilon = de;
						params.numLayers = 2;				params.numHidNeurons = new int[]{35};
						params.activFunctions = new int[]{MLPParameters.F_SIGMOID, MLPParameters.F_SIGMOID};
						// Run test
						CrossValidation cv = new CrossValidation(CROSS_VALID_K, params);
						cv.run(dataTrain);
						// Print results
						System.out.println(alphaOpts[a]+"\t"+miOpts[m]+"\t"+epsilonOpts[e]+"\t"+de+"\t"+cv.getAvgEE()+"\t"+cv.getAvgVE());
					}
				}
			}
		}
	}

	public static void test5(List<double[]> dataTrain){
		/* Test No.5: Optimal network parameters - fine-tunig:
		 * -> one chosen architecture: 1 hidden layer of 35 neurons, sig-sig activation functions
		 * -> alpha: 0.03-0.07
		 * -> mi: 0-0.1
		 * -> epsilon: 0.01-0.2
		 * -> dEpsilon: 0.98-0.93
		 * -> 250 epoch limit
		 */

		int count = 0; Date startTime = new Date();
		double[] alphaOpts = {0.03, 0.04, 0.05, 0.07};
		double[] miOpts = {0, 0.01, 0.1};
		double[] epsilonOpts = {0.01, 0.1, 0.2};
		double maxDEpsilon = 0.98, minDEpsilon = 0.92;
		System.out.println("alpha, mi, epsilon, dEpsilon, avg.est.err, avg.valid.err");
		for(int a = 0; a < alphaOpts.length; a++){
			for(int m = 0; m < miOpts.length; m++){
				for(int e = 0; e < epsilonOpts.length; e++){
					for(double de = maxDEpsilon; de > minDEpsilon; de -= 0.01){
						// Set parameters
						MLPParameters params = new MLPParameters();
						params.alpha = alphaOpts[a];		params.mi = miOpts[m];
						params.epsilon = epsilonOpts[e];	params.dEpsilon = de;
						params.numLayers = 2;				params.numHidNeurons = new int[]{35};
						params.activFunctions = new int[]{MLPParameters.F_SIGMOID, MLPParameters.F_SIGMOID};
						// Run test
						CrossValidation cv = new CrossValidation(CROSS_VALID_K, params);
						cv.run(dataTrain);
						// Print results
						System.out.println(alphaOpts[a]+"\t"+miOpts[m]+"\t"+epsilonOpts[e]+"\t"+de+"\t"+cv.getAvgEE()+"\t"+cv.getAvgVE());
						if(++count == 10){ System.out.println("TIME 10 = "+((new Date()).getTime() - startTime.getTime())); };
					}
				}
			}
		}
	}
	
	public static void testBest(List<double[]> dataTrain, List<double[]> dataTest){
		/* Test of best model:
		 * -> chosen architecture: 1 hidden layer of 35 neurons, sig-sig activation functions
		 * -> alpha: 0.04, mi: 0, epsilon: 0.01, dEpsilon: 0.98
		 * -> 1500 epoch limit
		 * -> report:
		 *    -> validation and test errors of all instances
		 *    -> average and best instance`s validation and test error
		 *    -> development of EE and VE of best instance through the process od training
		 *    -> best instance`s classification results for test data
		 *    -> best instance`s confusion matrix
		 */

		MLPParameters params = new MLPParameters();
		params.numLayers = 2;
		params.numHidNeurons = new int[]{35};
		params.activFunctions = new int[]{MLPParameters.F_SIGMOID, MLPParameters.F_SIGMOID};
		params.alpha = 0.04;
		params.mi = 0;
		params.epsilon = 0.01;
		params.dEpsilon = 0.98;
		params.maxEpoch = 1500;

		CrossValidation cv = new CrossValidation(CROSS_VALID_K, params);
		cv.run(dataTrain);
		CrossValidation.Result bestRes = cv.getBest();
		
		System.out.println("##### BEST MODEL TEST RESULTS #####");
		System.out.println("\n### Validation errors ###\nPer instance: [");
		for(int i = 0; i < cv.k; i++){
			System.out.println("\t" + cv.results[i].getFinalVErr());
		}
		System.out.println("]\nAverage: "+cv.getAvgVE());
		System.out.println("Best instance: "+bestRes.getFinalVErr());
		
		System.out.println("\n### Test errors ###\nPer instance: [");
		double avg = 0;
		for(int i = 0; i < cv.k; i++){
			double e = cv.results[i].mlp.test(dataTest);
			System.out.println("\t" + e);
			avg += e;
		}
		avg /= cv.k;
		System.out.println("]\nAverage: "+avg);
		System.out.println("Best instance: "+bestRes.mlp.test(dataTest));
		
		System.out.println("\n### Best model EE and VE development through training ###\nEE\tVE");
		double[] eErrors = bestRes.eErrors;
		double[] vErrors = bestRes.vErrors;
		for(int i = 0; i < eErrors.length; i++){
			System.out.println(eErrors[i]+"\t"+vErrors[i]);
		}
		
		System.out.println("\n### Best model 2D classification results ###");
		System.out.println("x, y, target, predicted");
		int[][] confusionMatrix = new int[CAT_NUM][CAT_NUM];
		for(int i = 0; i < dataTest.size(); i++){
			double[] data = dataTest.get(i);
			int target = (int)Math.round(data[N_INPUT]);
			int predict = bestRes.mlp.classify(dataTest, i);
			for(int j = 0; j < N_INPUT; j++){ System.out.print(data[j] +"\t"); }
			System.out.println(CAT_LABELS[target] +"\t"+ CAT_LABELS[predict]);
			confusionMatrix[target][predict]++;
		}

		
		System.out.println("\n### Best model confusion matrix ###");
		for(int i = 0; i < CAT_NUM; i++){ System.out.print("\t"+CAT_LABELS[i]); }
		System.out.println();
		for(int i = 0; i < CAT_NUM; i++){
			System.out.print(CAT_LABELS[i]);
			for(int j = 0; j < CAT_NUM; j++){
				System.out.print("\t"+confusionMatrix[i][j]);
			}
			System.out.println();
		}
	}
	
	public static List<double[]> readData(String filePath){
		ArrayList<double[]> data = new ArrayList<double[]>();
		try {
			Scanner in = new Scanner(new File(filePath));
			in.useLocale(Locale.US);
			while(in.hasNext()){
				double[] row = new double[N_INPUT+1];
				for(int i = 0; i < N_INPUT; i++){
					row[i] = in.nextDouble();
				}
				String label = in.next();
				for(int i = 0; i < CAT_NUM; i++){
					if(CAT_LABELS[i].equals(label)){
						row[N_INPUT] = (double)i;
						break;
					}
				}
				data.add(row);
			}
			in.close();
		}
		catch (FileNotFoundException e) { e.printStackTrace(); }
		return data;
	}
}