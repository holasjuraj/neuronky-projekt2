import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
	
	public final MLPParameters parameters;
	public final int k;
	public final Result[] results;

	public CrossValidation(int k, MLPParameters parameters) {
		this.k = k;
		this.parameters = parameters;
		results = new Result[k];
	}
	
	public Result[] run(List<double[]> data){
		final int size = data.size();
		@SuppressWarnings("unchecked")
		final List<double[]>[] parts = new List[k];
		for(int i = 0; i < k; i++){
			parts[i] = data.subList(
				(int)Math.round((double)(i*size)/(double)k),
				(int)Math.round((double)((i+1)*size)/(double)k));
		}
		
		// Start threads
		Thread[] threads = new Thread[Main.THREADS_NUM];
		for(int t = 0; t < Main.THREADS_NUM; t++){
			final int startPart = (int)Math.round((double)(t*k)/(double)Main.THREADS_NUM);
			final int endPart = (int)Math.round((double)((t+1)*k)/(double)Main.THREADS_NUM);
			Thread thread = new Thread(new Runnable() {
					public void run() {
						for(int vPartNum = startPart; vPartNum < endPart; vPartNum++){
							// Split into train and validation sets
							List<double[]> dataValid = new ArrayList<double[]>(size/k + 10);
							List<double[]> dataTrain = new ArrayList<double[]>((k-1)*size/k + 10);
							dataValid.addAll(parts[vPartNum]);
							for(int tPartNum = 0; tPartNum < k; tPartNum++){
								if(tPartNum == vPartNum){ continue; }
								dataTrain.addAll(parts[tPartNum]);
							}
							// Train model instance
							MLP mlp = new MLP(parameters);
							double[][] errors = mlp.trainAndValid(dataTrain, dataValid);
							Result res = new Result();
							res.mlp = mlp;
							res.eErrors = errors[0];
							res.vErrors = errors[1];
							results[vPartNum] = res;
						}
					}
				});
			threads[t] = thread;
			thread.start();
		}
		
		// Wait for threads
		for(int t = 0; t < Main.THREADS_NUM; t++){
			try { threads[t].join(); }
			catch (InterruptedException e) { System.out.println("Join thread "+t+" error"); e.printStackTrace(); }
		}
		
		return results;
	}
	
	public Result getBest(){
		Result best = results[0];
		for(int i = 1; i < k; i++){
			if(results[i].getFinalVErr() < best.getFinalVErr()){
				best = results[i];
			}
		}
		return best;
	}
	
	public double getAvgEE(){
		double sum = 0;
		for(int i = 0; i < k; i++){
			sum += results[i].getFinalEErr();
		}
		return sum / (double)k;
	}
	
	public double getAvgVE(){
		double sum = 0;
		for(int i = 0; i < k; i++){
			sum += results[i].getFinalVErr();
		}
		return sum / (double)k;
	}
	
	public static class Result{
		public MLP mlp;
		public double[] eErrors;
		public double[] vErrors;
		public double getFinalEErr(){ return eErrors[eErrors.length - 1]; }
		public double getFinalVErr(){ return vErrors[vErrors.length - 1]; }
	}

}