public class DiscretePerceptron extends Perceptron {

	public DiscretePerceptron(int n, double alpha) {
		super(n, alpha);
	}

	@Override
	protected double activFunc(double net) {
		if(net > 0){
			return 1;
		}
		return 0;
	}

	@Override
	protected double delta(double d, double y) {
		return d - y;
	}

	@Override
	protected double err(double d, double y) {
		return (d-y)*(d-y) / 2.0;
	}

}