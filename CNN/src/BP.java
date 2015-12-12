import java.io.*;
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
	private double inputError;
	private double hiddenError;
	private double outputError;

	private Random random;

	private double[] input; //input vector
	private double[] hidden; //hidden layer after sigmoid processed
	private double[] output; //output layer after sigmoid processed
	private double[] desire; //desired output

	private double[] deltaInput; //delta vector of the input layer
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

	private double[] sensitivityHidden; //sensitivities from input layer to hidden layer
	private double[] sensitivityOutput;  //sensitivities from hidden layer to output layer


	private int learningTimes = 10000; //learning times
	private double learningRate; //learning rate
	private double alpha = 0.9d;
	private double beta = 1.1d;


	public void setInput(int index, double value) {
		input[index] = value;
	}

	public void setDesire(int index, double value) {
		desire[index] = value;
	}

	public void reInitDesire() {
		desire = new double[outputNum];
	}


	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}

	public void setErrorLimit(double limit) {
		this.errorLimit = limit;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public void setLearningTimes(int times) {
		learningTimes = times;
	}

	public double getInputError(int index) {
		return deltaInput[index];
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

		deltaInput = new double[inputNum];
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

		sensitivityHidden = new double[hiddenNum];
		sensitivityOutput = new double[outputNum];

		random = new Random();
		randomizeBias(hiddenBias);
		randomizeBias(outputBias);
		randomizeWeights(weightInputHidden);
		randomizeWeights(weightHiddenOutput);

		this.learningRate = learningRate;

	}

	public void clearLayer() {
		for (int i = 0; i < hiddenNum; i++) {
			deltaHidden[i] = 0;
			hidden[i] = 0;
		}
		for (int i = 0; i < outputNum; i++) {
			output[i] = 0;
			deltaOutput[i] = 0;
		}
		for (int i = 0; i < hiddenNum; i++) {
			for (int j = 0; j < outputNum; j++) {
				deltaWeightHO[i][j] = 0;
			}
		}
		for (int i = 0; i < inputNum; i++) {
			for (int j = 0; j < hiddenNum; j++) {
				deltaWeightIH[i][j] = 0;
			}
		}
	}

	//randomize the weights from -1 to 1
	public void randomizeWeights(double[][] weights) {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				double randomDouble = (random.nextDouble() - 0.5d) * 2 * 2.36 / (Math.sqrt(inputNum));
//				double randomDouble = (random.nextDouble() - 0.5d) * 2 / weights.length;
				//weights[i][j] = random.nextDouble() > 0.5 ? randomDouble : -randomDouble;
				weights[i][j] = randomDouble;

			}
		}
	}

	//randomize the bias from -1 to 1
	public void randomizeBias(double[] bias) {
		for (int i = 0; i < bias.length; i++) {
//			double randomDouble = (random.nextDouble() - 0.5d) * 2 * 2.36 / (Math.sqrt(inputNum));
			double randomDouble = (random.nextDouble() - 0.5d) * 2;
			//bias[i] = random.nextDouble() > 0.5 ? randomDouble : -randomDouble;
			bias[i] = randomDouble;
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
			layerY[i] = sum;
		}

	}

	//calculate forward output
	public void forward() {
		layerForward(input, hidden, hiddenBias, weightInputHidden);
		ReLU(hidden);

		layerForward(hidden, output, outputBias, weightHiddenOutput);
		softmax(output);
	}


	//calculate the error of output layer
	public void calculateOutputError() {
		double errorSum = 0d;
		for (int i = 0; i < outputNum; i++) {
			double output_i = output[i];
			double result = desire[i] - output[i];
			sensitivityOutput[i] = result;
			deltaOutput[i] = result;
			errorSum += Math.abs(result);
		}

	}


	//calculate the error of hidden layer
	public void calculateHiddenError() {
		deltaHidden = new double[hiddenNum];
		double errorSum = 0d;
		for (int i = 0; i < hiddenNum; i++) {
			double outputErrorSum = 0d;
			for (int j = 0; j < outputNum; j++) {

				outputErrorSum += deltaOutput[j] * weightHiddenOutput[i][j];

			}

			deltaHidden[i] = outputErrorSum;
			//errorSum += Math.abs(result);

		}
		hiddenError = errorSum;
	}


	//calculate the error of input layer
	public void calculateInputError() {
		deltaInput = new double[inputNum];
		double errorSum = 0d;
		for (int i = 0; i < inputNum; i++) {
			double input_i = input[i];
			//double result;
			double hiddenErrorSum = 0d;
			for (int j = 0; j < hiddenNum; j++) {
				hiddenErrorSum += deltaHidden[j] * weightInputHidden[i][j];
			}
			//	result = hiddenErrorSum;
			deltaInput[i] = hiddenErrorSum;
			//	errorSum += Math.abs(result);

		}
		inputError = errorSum;
	}


	//calculate errors
	public void calculateError() {
		calculateOutputError();
		calculateHiddenError();
		calculateInputError();
	}


	//adjust the weight matrix between two layers
	public void adjustWeights(double[] layer, double[] deltaError, double[][] weight, double[][] deltaWeight) {
		double delta = 0d;
		for (int i = 0; i < weight.length; i++) {
			for (int j = 0; j < weight[i].length; j++) {
				delta = learningRate * deltaError[j] * layer[i];
				deltaWeight[i][j] = delta;
				weight[i][j] += deltaWeight[i][j];
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
		calculateOutputError();
		calculateHiddenError();
		adjustWeights(hidden, deltaOutput, weightHiddenOutput, deltaWeightHO);
		adjustBias(outputBias, deltaOutputBias);

		dReLU(hidden);
		for (int i = 0; i < hiddenNum; i++) {
			deltaHidden[i] *= hidden[i];
		}
		calculateInputError();
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


	//get the desire result
	public int getDesire() {
		int index = 0;
		for (int i = 0; i < outputNum; i++) {
			if (desire[i] == 1) {
				index = i;
			}
		}
		return index;
	}


	//the sigmoid function
	public double sigmoid(double x) {
		return 1d / (1d + Math.exp(-x));
	}


	//the ReLu function
	public double ReLU(double x) {
		if (x > 0) {
			return x;
		} else {
			return 0;
		}
	}


	//the ReLu function for an array
	public void ReLU(double[] x) {
		int length = x.length;
		for (int i = 0; i < length; i++) {
			x[i] = ReLU(x[i]);
		}
	}


	//the derivative of ReLU function
	public double dReLU(double x) {
		if (x > 0) {
			return 1;
		} else {
			return 0;
		}
	}

	//the derivation of ReLU function for an array
	public void dReLU(double[] x) {
		int length = x.length;
		for (int i = 0; i < length; i++) {
			x[i] = dReLU(x[i]);
		}
	}

	//the softmax function
	public void softmax(double[] x) {
		double sum = 0d;
		double max = -1000;
		for (int i = 0; i < x.length; i++) {
			if (max < x[i]) {
				max = x[i];
			}
		}
		for (int i = 0; i < x.length; i++) {
			sum += Math.exp(x[i] - max);
		}
		for (int i = 0; i < x.length; i++) {
			x[i] = Math.exp(x[i] - max) / sum;
		}
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

