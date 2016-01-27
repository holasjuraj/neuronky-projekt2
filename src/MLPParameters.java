import org.ejml.simple.SimpleMatrix;

public class MLPParameters {
	public static final int F_SIGMOID = 0;
	public static final int F_TANH = 1;
	public static final int F_LINEAR = 2;
	public static final int F_SMOOTH_R_LIN = 3;
	
	public double alpha = 0.1;		// Learning rate
	public double mi = 0;			// Momentum
	public double epsilon = 0;		// Weight decay factor
	public double dEpsilon = 1;		// Decay factor of epsilon
	public int maxEpoch = 250;
	public int numLayers = 2;		// Including output layer
	public int[] numHidNeurons = {20};		// 1st hid.l. , ... , n-th hid.l.
	public int[] activFunctions = {F_SIGMOID, F_SIGMOID};	// 1st hid.l. , ... , n-th hid.l. , output l.

	public SimpleMatrix activ(int layer, SimpleMatrix net, boolean addBiasToRes){
		int rows = net.numRows(), cols = net.numCols();
		if(addBiasToRes){ rows++; }
		SimpleMatrix result = new SimpleMatrix(rows, cols);
		// Shortcut for linear function
		if(activFunctions[layer] == F_LINEAR){
			result.insertIntoThis(0, 0, net);
			if(addBiasToRes){ result.set(rows-1, 0, -1); }
			return result;
		}
		// Map other functions
		for(int r = 0; r < rows; r++){
			for(int c = 0; c < cols; c++){
				double x = -1;
				if(!addBiasToRes || r<rows-1){
					// Choose activation function
					switch(activFunctions[layer]){
						case F_SIGMOID:	x = sigmoid(net.get(r,c)); break;
						case F_TANH:	x = Math.tanh(net.get(r,c)); break;
						case F_SMOOTH_R_LIN: x = Math.log1p(Math.exp(net.get(r,c))); break;
					}
				}
				result.set(r, c, x);
			}
		}
		return result;
	}
	
	public SimpleMatrix activDer(int layer, SimpleMatrix net){
		int rows = net.numRows(), cols = net.numCols();
		SimpleMatrix result = new SimpleMatrix(rows, cols);
		// Shortcut for linear function
		if(activFunctions[layer] == F_LINEAR){
			return result.plus(1);
		}
		// Map other functions
		for(int r = 0; r < rows; r++){
			for(int c = 0; c < cols; c++){
				double x = -1.0;
				// Choose activation function
				switch(activFunctions[layer]){
					case F_SIGMOID:	x = sigmoidDer(net.get(r,c)); break;
					case F_TANH:	x = tanhDer(net.get(r, c)); break;
					case F_SMOOTH_R_LIN: x = smoothDer(net.get(r, c)); break;
				}
				result.set(r, c, x);
			}
		}
		return result;
	}
	
	private double sigmoid(double net){
		return 1 / (1 + Math.exp(-net));
	}
	
	private double sigmoidDer(double net){
		double fNet = sigmoid(net);
		return fNet*(1-fNet);
	}
	
	private double tanhDer(double net){
		double cosh = Math.cosh(net);
		double cosh2x1 = Math.cosh(2*net) + 1;
		return (4*cosh*cosh) / (cosh2x1*cosh2x1);
	}
	
	private double smoothDer(double net){
		double eNet = Math.exp(net);
		return eNet / (eNet + 1);
	}
	
}