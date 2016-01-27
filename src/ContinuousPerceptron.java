public class ContinuousPerceptron extends Perceptron {

	public ContinuousPerceptron(int n, double alpha) {
		super(n, alpha);
	}

	@Override
	protected double activFunc(double net) {
		return 1.0 / (1.0 + Math.exp(-net));
	}

	@Override
	protected double delta(double d, double y) {
		return (d-y) * y * (1.0-y);
	}

	@Override
	protected double err(double d, double y) {
		return (d-y)*(d-y) / 2.0;
	}

}