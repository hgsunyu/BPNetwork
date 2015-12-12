import java.io.Serializable;
import java.util.Random;

/**
 * Created by W on 2015/10/3.
 */
public class BP implements Serializable {
	private int inputNum; //input layer number
	private int hiddenNum; //hidden layer number
	private int outputNum; //output layer number

	private double totalError = 1.0d; //total error
	private double errorLimit = 0.01d; //error limit
	private double hiddenError;
	private double outputError;

	private Random random;

	private double[] input; //input vector
	private double[] hidden; //hidden layer after sigmoid processed
	private double[] output; //output layer after sigmoid processed
	private double[] desire; //desired output

	private double[] deltaHidden; //delta vector of the hidden layer
	private double[] deltaOutput; //delta vector of the output layer

	private double[] hiddenBias; //the bias of hidden units
	private double[] outputBias; //the bias of output units

	private double[] deltaHiddenBias; //delta vector of the bias of hidden units
	private double[] deltaOutputBias; //delta vector of the bias of output units

	private double[][] weightInputHidden; //weight matrix from input layer to hidden layer
	private double[][] weightHiddenOutput; //weight matrix from hidden layer to output layer

	private double[][] deltaWeightIH; //delta weight matrix from input layer to hidden layer
	private double[][] deltaWeightHO; //delta weight matrix from hidden layer to output layer


	private int learningTimes = 10000; //learning times
	private double learningRate; //learning rate
	private double alpha = 0.9d;
	private double beta = 1.1d;


	public void setInput(int index,double value){
		input[index]=value;
	}

	public void setDesire(int index,double value){
		desire[index]=value;
	}

	public void reInitDesire(){
		desire=new double[outputNum];
	}


	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}

	public void setErrorLimit(double limit){
		this.errorLimit=limit;
	}

	public void setAlpha(double alpha){
		this.alpha=alpha;
	}

	public void setBeta(double beta){
		this.beta=beta;
	}

	public void setLearningTimes(int times){
		learningTimes=times;
	}



	//constructor
	public BP(int inputNum, int hiddenNum, int outputNum, double learningRate) {

		this.inputNum = inputNum;
		this.hiddenNum = hiddenNum;
		this.outputNum = outputNum;
		this.learningRate = learningRate;


		input = new double[inputNum];
		hidden = new double[hiddenNum];
		output = new double[outputNum];
		desire = new double[outputNum];

		deltaHidden = new double[hiddenNum];
		deltaOutput = new double[outputNum];

		hiddenBias = new double[hiddenNum];
		outputBias = new double[outputNum];

		deltaHiddenBias = new double[hiddenNum];
		deltaOutputBias = new double[outputNum];

		weightInputHidden = new double[inputNum][hiddenNum];
		weightHiddenOutput = new double[hiddenNum][outputNum];

		deltaWeightIH = new double[inputNum][hiddenNum];
		deltaWeightHO = new double[hiddenNum][outputNum];

		random = new Random();
		randomizeBias(hiddenBias);
		randomizeBias(outputBias);
		randomizeWeights(weightInputHidden);
		randomizeWeights(weightHiddenOutput);

		this.learningRate = learningRate;

	}




	//randomize the weights from -1 to 1
	public void randomizeWeights(double[][] weights) {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				double randomDouble = (random.nextDouble() - 0.5d) * 2 * 2.36 / (Math.sqrt(inputNum));
				weights[i][j] = random.nextDouble() > 0.5 ? randomDouble : -randomDouble;
			}
		}
	}

	//randomize the bias from -1 to 1
	public void randomizeBias(double[] bias) {
		for (int i = 0; i < bias.length; i++) {
			double randomDouble = (random.nextDouble() - 0.5d) * 2 * 2.36 / (Math.sqrt(inputNum));
			bias[i] = random.nextDouble() > 0.5 ? randomDouble : -randomDouble;
		}
	}

	//calculate forward output betweeen layerX and layerY
	public void layerForward(double[] layerX, double[] layerY, double[] layerYBias, double[][] weightXY) {
		for (int i = 0; i < layerY.length; i++) {
			double sum = 0d;
			for (int j = 0; j < layerX.length; j++) {
				sum += weightXY[j][i] * layerX[j];
			}
			sum += layerYBias[i];
			layerY[i] = sigmoid(sum);
		}

	}

	//calculate forward output
	public void forward() {
		layerForward(input, hidden, hiddenBias, weightInputHidden);
		layerForward(hidden, output, outputBias, weightHiddenOutput);
	}

	//calculate the error of output layer
	public void calculateOutputError() {
		double errorSum = 0d;
		for (int i = 0; i < outputNum; i++) {
			double output_i = output[i];
			double result;
			result = output_i * (1d - output_i) * (desire[i] - output_i);
			deltaOutput[i] = result;
			errorSum += Math.abs(result);
		}
		outputError = errorSum;
	}


	//calculate the error of hidden layer
	public void calculateHiddenError() {
		double errorSum = 0d;
		for (int i = hiddenNum - 1; i > 0; i--) {
			double hidden_i = hidden[i];
			double result;
			double outputErrorSum = 0d;
			for (int j = 0; j < outputNum; j++) {
				outputErrorSum += deltaOutput[j] * weightHiddenOutput[i][j];
			}
			result = hidden_i * (1d - hidden_i) * outputErrorSum;
			deltaHidden[i] = result;
			errorSum += Math.abs(result);
		}
		hiddenError = errorSum;
	}


	public void calculateError() {
		calculateOutputError();
		calculateHiddenError();

	}


	//calculate the total error: 1/2*sigma(di-oi)^2
	public double calculateTotalError() {
		double error = 0d;
		for (int i = 0; i < outputNum; i++) {
			error += Math.pow(desire[i] - output[i], 2);
		}
		error *= 0.5d;
		return error;
	}



	//adjust the weight matrix between two layers
	public void adjustWeights(double[] layer, double[] deltaError, double[][] weight, double[][] deltaWeight) {
		double delta = 0d;
		for (int i = 0; i < weight.length; i++) {
			for (int j = 0; j < weight[i].length; j++) {
				delta = learningRate * deltaError[j] * layer[i];
				deltaWeight[i][j] = delta;
				weight[i][j] += delta;
			}
		}


	}


	//adjust the bias matrix between two layers
	public void adjustBias(double[] bias, double[] deltaBias) {
		double delta = 0d;
		for (int i = 0; i < bias.length; i++) {
			delta = learningRate * deltaBias[i];
			deltaBias[i] = delta;
			bias[i] += delta;
		}

	}


	//adjust the weights matrix and the bias matrix
	public void backward() {
		adjustWeights(hidden, deltaOutput, weightHiddenOutput, deltaWeightHO);
		adjustBias(outputBias, deltaOutputBias);
		adjustWeights(input, deltaHidden, weightInputHidden, deltaWeightIH);
		adjustBias(hiddenBias, deltaHiddenBias);

	}


	//check if the output correct or not
	public boolean isOutputCorrect() {
		double max = output[0];
		int max_index = 0;
		for (int i = 1; i < outputNum; i++) {
			if (output[i] > max) {
				max = output[i];
				max_index = i;
			}
		}
		return desire[max_index] == 1 ? true : false;
	}


	//the sigmoid function
	public double sigmoid(double x) {
		return 1d / (1d + Math.exp(-x));
	}


	//get the result of a test sample
	public int getOutput() {
		double max = output[0];
		int max_index = 0;
		for (int i = 1; i < outputNum; i++) {
			if (output[i] > max) {
				max = output[i];
				max_index = i;
			}
		}

		return max_index;
	}



}

