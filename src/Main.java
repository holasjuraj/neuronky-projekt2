import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

public class Main {

	public static final int N = 3;
	public static final long MAX_EPOCH_NUM = 1000000;
	public static final int AVG_ITERATIONS = 1000;
	public static final double ALPHA_LIMIT = 1;
	public static final double ALPHA_STEP = 0.01;
	
	public static void main(String[] args) {
		// Uncomment separate sections for testing
		
		// Prepare dataset
		List<Entry< List<Double>, Double >> data = makeAndData(N);
		//List<Entry< List<Double>, Double >> data = makeOrData(N);
		
		// Test convergence
		/**/
		Perceptron per = new ContinuousPerceptron(N, 0);
		for(double alpha = ALPHA_STEP; alpha < ALPHA_LIMIT+ALPHA_STEP; alpha += ALPHA_STEP){
			per.alpha = alpha;
			double avg = 0;
			for(int i = 0; i < AVG_ITERATIONS; i++){
				avg += trainData(per, data, false);
			}
			avg /= AVG_ITERATIONS;
			System.out.println(avg);
		}
		/**/
		
		// Report one training
		/*/
		trainData(new ContinuousPerceptron(N, 0.2), data, true);
		/**/
		
		// Compare with discrete perceptron
		/*/
		Perceptron perCon = new ContinuousPerceptron(N, 0);
		Perceptron perDis = new DiscretePerceptron(N, 0);
		for(double alpha = ALPHA_STEP; alpha < ALPHA_LIMIT+ALPHA_STEP; alpha += ALPHA_STEP){
			perCon.alpha = alpha;
			perDis.alpha = alpha;
			double avgCon = 0, avgDis = 0;
			for(int i = 0; i < AVG_ITERATIONS; i++){
				avgCon += trainData(perCon, data, false);
				avgDis += trainData(perDis, data, false);
			}
			avgCon /= AVG_ITERATIONS;
			avgDis /= AVG_ITERATIONS;
			System.out.println(avgDis / avgCon);
		}
		/**/
	}
	
	private static long trainData(Perceptron perceptron, List<Entry< List<Double>, Double >> data, boolean print){
		perceptron.reset();
		long epNum = 0;
		double E = 1;
		while(E > 0){
			epNum++;
			if(print){ System.out.println("Epoch "+epNum+":"); }
			E = perceptron.trainEpoch(data, print);
			
			if(print){
				System.out.println(	"  E = "+E+"\n  w = [");
				for(int j = 0; j < perceptron.w.size(); j++){
					System.out.println("       "+perceptron.w.get(j));
				}
				System.out.println("      ]\n");
			}
			if(epNum >= MAX_EPOCH_NUM){ break; }
		}
		
		if(print){
			System.out.println(	"\nRESULT:"+
								"\n  number of epochs: "+epNum+
								"\n  w = [");
			for(int j = 0; j < perceptron.w.size(); j++){
				System.out.println("       "+perceptron.w.get(j));
			}
			System.out.println("      ]");
		}
		
		return epNum;
	}
	
	private static List<List<Double>> generateBits(int n){
		int count = 1 << n;
		List<List<Double>> result = new ArrayList<List<Double>>(count);
		for(int i = 0; i < count; i++){
			List<Double> p = new ArrayList<>(n);
			for(int j = n-1; j >= 0; j--){
				p.add((double)( (i >>> j) & 1 ));
			}
			result.add(p);
		}
		return result;
	}
	
	private static List<Entry< List<Double>, Double >> makeAndData(int n){
		int count = 1 << n;
		List<Entry< List<Double>, Double >> result = new ArrayList<Entry< List<Double>, Double >>(count);
		List<List<Double>> bits = generateBits(n);
		for(int i = 0; i < count; i++){
			double d = 1;
			for(int j = 0; j < n; j++){
				d *= bits.get(i).get(j);
			}
			result.add(new AbstractMap.SimpleEntry<List<Double>, Double>(
							bits.get(i), d
						));
		}
		return result;
	}
	
	private static List<Entry< List<Double>, Double >> makeOrData(int n){
		int count = 1 << n;
		List<Entry< List<Double>, Double >> result = new ArrayList<Entry< List<Double>, Double >>(count);
		List<List<Double>> bits = generateBits(n);
		for(int i = 0; i < count; i++){
			double d = 0;
			for(int j = 0; j < n; j++){
				d += bits.get(i).get(j);
			}
			if(d > 0){ d = 1.0; }
			result.add(new AbstractMap.SimpleEntry<List<Double>, Double>(
							bits.get(i), d
						));
		}
		return result;
	}

}