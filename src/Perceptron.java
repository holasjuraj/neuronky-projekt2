import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.Random;

public abstract class Perceptron {
	
	List<Double> w;
	double alpha;
	
	public Perceptron(int n, double alpha){
		w = new ArrayList<Double>(n+1);
		Random rand = new Random();
		for(int i = 0; i < n+1; i++){
			w.add(rand.nextDouble());
		}
		this.alpha = alpha;
	}
	
	public void reset(){
		Random rand = new Random();
		for(int i = 0; i < w.size(); i++){
			w.set(i, rand.nextDouble());
		}
	}
	
	public double trainEpoch(List<Entry< List<Double>, Double >> data, boolean print){
		double E = 0;
		Collections.shuffle(data);
		for(int i = 0; i < data.size(); i++){
			List<Double> x = data.get(i).getKey();
			x.add(-1.0);	// Add bias
			double d = data.get(i).getValue();
			double y = output(x);
			double e = err(d, Math.round(y));
			
			if(e > 0){
				for(int j = 0; j < w.size(); j++){
					w.set(j, w.get(j) + alpha*delta(d, y)*x.get(j) );
				}
			}
			
			E += e;
			x.remove(x.size()-1);	// Remove bias from data
			
			if(print){
				System.out.print("  x =");
				for(int j = 0; j < x.size(); j++){
					System.out.print(" "+Math.round(x.get(j)));
				}
				System.out.println(" | d = "+Math.round(d)+" | y = "+Math.round(y)+((e > 0) ? " !" : ""));
			}
		}
		return E;
	}
	
	public double output(List<Double> x){
		double net = 0;
		for(int i = 0; i < x.size(); i++){
			net += x.get(i) * w.get(i);
		}
		return activFunc(net);
	}
	
	protected abstract double activFunc(double net);
	
	protected abstract double delta(double d, double y);
	
	protected abstract double err(double d, double y);

}