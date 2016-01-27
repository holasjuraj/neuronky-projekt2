import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class MLP {
	public static final int N_INPUT = Main.N_INPUT;
	public static final int CAT_NUM = Main.CAT_NUM;

	private MLPParameters prm;
	private SimpleMatrix[] w;
	private double[] means;

	public MLP(MLPParameters parameters) {
		prm = parameters;
	}
	
	public void reset(){
		means = new double[N_INPUT];
		Random rand = new Random();
		// Initializing weight matrixes
		int sizes[] = new int[prm.numLayers+1];
		sizes[0] = N_INPUT;
		for(int i = 1; i < prm.numLayers; i++){
			sizes[i] = prm.numHidNeurons[i-1];
		}
		sizes[prm.numLayers] = CAT_NUM;
		w = new SimpleMatrix[prm.numLayers];
		for(int i = 0; i < prm.numLayers; i++){
			w[i] = SimpleMatrix.random(sizes[i+1], sizes[i]+1, 0, 1, rand);
		}
	}
	
	public double[][] trainAndValid(List<double[]> dataTrain, List<double[]> dataValid){
		reset();
		double[] eErrors = new double[prm.maxEpoch+1];
		double[] vErrors = new double[prm.maxEpoch+1];
		
		// Mean removal
		for(int r = 0; r < dataTrain.size(); r++){
			for(int c = 0; c < N_INPUT; c++){
				means[c] += dataTrain.get(r)[c];
			}
		}
		for(int c = 0; c < N_INPUT; c++){
			means[c] /= dataTrain.size();
		}
		
		// Training and testing
		double minValErr = 1;
		SimpleMatrix bestW[] = new SimpleMatrix[prm.numLayers];
		SimpleMatrix[] oldDeltaW = new SimpleMatrix[prm.numLayers];
		for(int i = 0; i < prm.numLayers; i++){
			bestW[i] = new SimpleMatrix(w[i].numRows(), w[i].numCols());
			oldDeltaW[i] = new SimpleMatrix(w[i].numRows(), w[i].numCols());
		}
		
		for(int epNum = 0; epNum < prm.maxEpoch; epNum++){
			Collections.shuffle(dataTrain);
		
			// Train epoch
			for(int j = 0; j < dataTrain.size(); j++){
				// Forward pass
				SimpleMatrix x = getInput(dataTrain, j, means);
				SimpleMatrix[][] fpResult = forwardPass(x);
				// Backward pass
				SimpleMatrix d = getTarget(dataTrain, j);
				SimpleMatrix[] DeltaW = backwardPass(d, fpResult);
				// Adjust weights
				for(int i = 0; i < prm.numLayers; i++){
					w[i] = w[i].plus(DeltaW[i]).plus(oldDeltaW[i].scale(prm.mi)).minus(w[i].scale(prm.epsilon));
					oldDeltaW[i] = DeltaW[i];
				}
				prm.epsilon *= prm.dEpsilon;
			}
			
			// Epoch estimation error
			double eErr = test(dataTrain);
			eErrors[epNum] = eErr;
			if(eErr < minValErr){
				minValErr = eErr;
				for(int i = 0; i < prm.numLayers; i++){
					bestW[i].set(w[i]);
				}
			}
			
			// Epoch validation error
			vErrors[epNum] = test(dataValid);
		}

		// Revert to minimal error state
		for(int i = 0; i < prm.numLayers; i++){
			w[i] = bestW[i];
		}
		eErrors[prm.maxEpoch] = test(dataTrain);
		vErrors[prm.maxEpoch] = test(dataValid);
		
		return new double[][] {eErrors, vErrors};
	}
	
	public double test(List<double[]> data){
		double E = 0;
		for(int j = 0; j < data.size(); j++){
			int target = (int)Math.round(data.get(j)[N_INPUT]);
			if(target != classify(data, j)){ E++; }
		}
		E = E / (double)data.size();
		return E;
	}
	
	public int classify(List<double[]> data, int index){
		SimpleMatrix
			x = getInput(data, index, means),
			y = forwardPass(x)[0][prm.numLayers];
		return outputToClass(y);
	}
	
	private SimpleMatrix[][] forwardPass(SimpleMatrix x){
		SimpleMatrix[]
			xs = new SimpleMatrix[prm.numLayers+1],
			nets = new SimpleMatrix[prm.numLayers];
		xs[0] = x;
		for(int i = 0; i < prm.numLayers; i++){
			nets[i] = w[i].mult(xs[i]);
			xs[i+1] = prm.activ(i, nets[i], i<prm.numLayers-1);
		}
		return new SimpleMatrix[][] {xs, nets};
	}
	
	
	private SimpleMatrix[] backwardPass(SimpleMatrix d, SimpleMatrix[][] forwardPassResult){
		SimpleMatrix[]
			xs = forwardPassResult[0],
			nets = forwardPassResult[1],
			DeltaW = new SimpleMatrix[prm.numLayers];
		SimpleMatrix
			y = xs[prm.numLayers],
			layerOutput = d.minus(y);
		for(int i = prm.numLayers-1; i >= 0; i--){
			SimpleMatrix delta = layerOutput.elementMult(prm.activDer(i, nets[i]));
			DeltaW[i] = delta.mult(xs[i].transpose()).scale(prm.alpha);
			if(i > 0){
				SimpleMatrix wUnbias = w[i].extractMatrix(0, w[i].numRows(), 0, w[i].numCols()-1);
				layerOutput = wUnbias.transpose().mult(delta);
			}			
		}
		return DeltaW;
	}
	
	
	private static SimpleMatrix getInput(List<double[]> data, int index, double[] means){
		SimpleMatrix X = new SimpleMatrix(N_INPUT+1, 1);
		for(int i = 0; i < N_INPUT; i++){
			X.set(i, 0, data.get(index)[i] - means[i]);
		}
		X.set(N_INPUT, 0, -1.0);
		return X;
	}
	
	private static SimpleMatrix getTarget(List<double[]> data, int index){
		SimpleMatrix Y = new SimpleMatrix(CAT_NUM, 1);
		Y.set((int)Math.round(data.get(index)[N_INPUT]), 0, 1.0);
		return Y;
	}
	
	private static int outputToClass(SimpleMatrix y){
		int maxIndex = 0;
		double max = Double.MIN_VALUE;
		for(int i = 0; i < y.numRows(); i++){
			if(y.get(i, 0) > max){
				max = y.get(i, 0);
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
}