import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;

/**
 * CNN:Convolutional Neural Network
 * Created by W on 2015/10/24.
 */
public class CNN implements Serializable {

	private DecimalFormat df = new DecimalFormat("0.00");

	public BP bp;//BP network

	private int inputNum; //input layer number
	private int outputNum;//output layer number

	private double[][] input; //input vector
	private double[] output;  //output vector
	private double[] desire;  //desire vector

	//C:convolution layer; S:subsampling layer
	private double[][][] C1;
	private double[][][] S2;
	private double[][][] C3;
	private double[][][] S4;
	private double[] C5;
	private double[] F6;

	private double[][][] kernelC1;
	private double[][][] kernelC3;
	private double[][][] kernelC5;
	private double[] biasC1;
	private double[] biasS2;
	private double[] biasC3;
	private double[] biasS4;
	private double[] biasC5;

	//sensitivities
	private double[][] sensitivityInput;
	private double[][][] sensitivityC1;
	private double[][][] sensitivityS2;
	private double[][][] sensitivityC3;
	private double[][][] sensitivityS4;
	private double[][][] sensitivityC5;


	//hyper parameters:
	private int featureMapNumC1;
	private int featureMapNumC3;
	private int featureMapNumC5;
	private int kernelNumC1;
	private int kernelNumC3;
	private int kernelNumC5;
	private int kernelSize;
	private int stride;
	private double learningRate;
	private double regularizationRate;


	private Random random;


	//temp
	public double[][] getInput() {
		return input;
	}


	public void setInput(int x, int y, double value) {
		input[x][y] = value;
	}

	public void setDesire(int index, double value) {
		desire[index] = value;
	}

	public void reInitDesire() {
		desire = new double[outputNum];
	}

	public int getFeatureMapNumC1() {
		return featureMapNumC1;
	}

	public void setFeatureMapNumC1(int featureMapNumC1) {
		this.featureMapNumC1 = featureMapNumC1;
	}

	public int getFeatureMapNumC3() {
		return featureMapNumC3;
	}

	public void setFeatureMapNumC3(int featureMapNumC3) {
		this.featureMapNumC3 = featureMapNumC3;
	}

	public int getKernelNumC1() {
		return kernelNumC1;
	}

	public void setKernelNumC1(int kernelNumC1) {
		this.kernelNumC1 = kernelNumC1;
	}

	public int getKernelNumC3() {
		return kernelNumC3;
	}

	public void setKernelNumC3(int kernelNumC3) {
		this.kernelNumC3 = kernelNumC3;
	}

	public int getKernelSize() {
		return kernelSize;
	}

	public void setKernelSize(int kernelSize) {
		this.kernelSize = kernelSize;
	}

	public int getStride() {
		return stride;
	}

	public void setStride(int stride) {
		this.stride = stride;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getRegularizationRate() {
		return regularizationRate;
	}

	public void setRegularizationRate(double regularizationRate) {
		this.regularizationRate = regularizationRate;
	}

	public int getFeatureMapNumC5() {
		return featureMapNumC5;
	}

	public void setFeatureMapNumC5(int featureMapNumC5) {
		this.featureMapNumC5 = featureMapNumC5;
	}

	public int getKernelNumC5() {
		return kernelNumC5;
	}

	public void setKernelNumC5(int kernelNumC5) {
		this.kernelNumC5 = kernelNumC5;
	}

	//constructor
	public CNN(int inputNum, int outputNum, int featureMapNumC1, int featureMapNumC3, int featureMapNumC5, int kernelSize, int stride, double learningRate, double regularizationRate) {
		this.inputNum = inputNum;
		this.outputNum = outputNum;
		this.featureMapNumC1 = featureMapNumC1;
		this.featureMapNumC3 = featureMapNumC3;
		this.featureMapNumC5 = featureMapNumC5;
		this.kernelNumC1 = featureMapNumC1;
		this.kernelNumC3 = featureMapNumC1 * featureMapNumC3;
		this.kernelNumC5 = featureMapNumC3 * featureMapNumC5;
		this.kernelSize = kernelSize;
		this.stride = stride;
		this.learningRate = learningRate;
		this.regularizationRate = regularizationRate;

		input = new double[inputNum][inputNum];
		output = new double[outputNum];
		desire = new double[outputNum];

		kernelC1 = new double[kernelNumC1][kernelSize][kernelSize];
		kernelC3 = new double[kernelNumC3][kernelSize][kernelSize];
		kernelC5 = new double[kernelNumC5][kernelSize][kernelSize];
		biasC1 = new double[featureMapNumC1];
		biasS2 = new double[featureMapNumC1];
		biasC3 = new double[featureMapNumC3];
		biasS4 = new double[featureMapNumC3];
		biasC5 = new double[featureMapNumC5];

		C1 = new double[featureMapNumC1][][];
		S2 = new double[featureMapNumC1][][];
		C3 = new double[featureMapNumC3][][];
		S4 = new double[featureMapNumC3][][];
		C5 = new double[featureMapNumC5];


		sensitivityInput = new double[inputNum][inputNum];
		sensitivityC1 = new double[featureMapNumC1][][];
		sensitivityS2 = new double[featureMapNumC1][][];
		sensitivityC3 = new double[featureMapNumC3][][];
		sensitivityS4 = new double[featureMapNumC3][][];
		sensitivityC5 = new double[featureMapNumC5][1][1];


		random = new Random();

		randomizeKernel(kernelC1, kernelNumC1, kernelSize);
		randomizeKernel(kernelC3, kernelNumC3, kernelSize);
		randomizeKernel(kernelC5, kernelNumC5, kernelSize);
		randomizeBias(biasC1, featureMapNumC1);
		randomizeBias(biasS2, featureMapNumC1);
		randomizeBias(biasC3, featureMapNumC3);
		randomizeBias(biasS4, featureMapNumC3);
		randomizeBias(biasC5, featureMapNumC5);

		bp = new BP(120, 84, 10, 0.008);

	}

	//reinit matrix
	public void reInitMatrix(double[][][] matrix) {
		matrix = new double[matrix.length][matrix[0].length][matrix[0].length];
	}

	public void reInitMatrix(double[][] matrix) {
		matrix = new double[matrix.length][matrix[0].length];
	}

	public void reInitMatrix(double[] matrix) {
		matrix = new double[matrix.length];
	}


	//randomize kernel
	public void randomizeKernel(double[][][] kernel, int kernelNum, int kernelSize) {
		for (int i = 0; i < kernelNum; i++) {
			for (int j = 0; j < kernelSize; j++) {
				for (int k = 0; k < kernelSize; k++) {
					double randomDouble = (random.nextDouble() - 0.5d) * 2 * 2.36 / (inputNum);
//					double randomDouble = (random.nextDouble() - 0.5d) * 2 / (inputNum);
					kernel[i][j][k] = randomDouble;
				}
			}
		}
	}


	//ranomize bias
	public void randomizeBias(double[] bias, int mapNum) {
		for (int i = 0; i < mapNum; i++) {
//			double randomDouble = (random.nextDouble() - 0.5d) * 2 * 2.36 / (inputNum);
			double randomDouble = (random.nextDouble() - 0.5d) * 2;
			bias[i] = randomDouble;
		}
	}


	//get the size of a new after-convolution matrix
	public int getNewMapSize(int matrixSize, int kernelSize, int stride) {
		int result = (int) ((double) ((matrixSize - kernelSize) / stride) + 1);
		return result;
	}


	//convolution between two matrix that have the same sizes
	public double convolution(double[][] oldMap, double[][] kernel) {
		int size = kernel.length;

		double sum = 0d;
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				sum += oldMap[x][y] * kernel[x][y];
			}
		}
		return sum;

	}

    //convolution between two matrix that have the same sizes
	public double convolution(double[][] oldMap, double[][] kernel, double bias) {
		int size = kernel.length;

		double sum = 0d;
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				sum += oldMap[x][y] * kernel[x][y];
			}
		}
		double result = sum;
		return result;

	}


	//convolution with adding bias
	public void convolution(double[][] oldMap, double[][] kernel, double bias, double[][] newMap) {
		int size = newMap.length;
		int deltaX = 0, deltaY = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				double sum = 0d;
				for (int x = 0; x < kernel.length; x++) {
					for (int y = 0; y < kernel.length; y++) {
						sum += oldMap[x + deltaX][y + deltaY] * kernel[x][y];
					}
				}
				double result = sum + bias;
				newMap[i][j] = result;

				deltaY += stride;
			}
			deltaX += stride;
			deltaY = 0;
		}
	}


	//convolution without adding bias
	public void convolution(double[][] oldMap, double[][] kernel, double[][] newMap) {
		int size = newMap.length;
		int deltaX = 0, deltaY = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				double sum = 0d;
				for (int x = 0; x < kernel.length; x++) {
					for (int y = 0; y < kernel.length; y++) {
//						sum += oldMap[x + deltaX][y + deltaY] * kernel[x][y];
						sum += oldMap[x + i][y + i] * kernel[x][y];
					}
				}
				newMap[i][j] = sum;
				deltaY += stride;

			}
			deltaX += stride;
			deltaY = 0;
		}

	}


	//sub-sampling
	public void meanPooling(double[][] oldMap, double[][] newMap, double bias) {
		int size = newMap.length;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				double sum = oldMap[i * 2][j * 2] + oldMap[i * 2][j * 2 + 1] + oldMap[i * 2 + 1][j * 2] + oldMap[i * 2 + 1][j * 2 + 1];
				double result = sum / 4;
				result = result + bias;
				newMap[i][j] = result;
			}
		}

	}


	//forward
	public void forward() {
		for (int i = 0; i < featureMapNumC1; i++) {
			int sizeC = getNewMapSize(inputNum, kernelSize, stride);
			C1[i] = new double[sizeC][sizeC];

			//input to C1: convolution
			convolution(input, kernelC1[i], biasC1[i], C1[i]);

			//C1 to S2: mean pooling
			int sizeS = (int) ((double) sizeC / 2);
			S2[i] = new double[sizeS][sizeS];
			meanPooling(C1[i], S2[i], biasS2[i]);
		}


		for (int i = 0; i < featureMapNumC3; i++) {
			int sizeC = getNewMapSize(S2[0].length, kernelSize, stride);
			C3[i] = new double[sizeC][sizeC];

			//S2 to C3: convolution
			for (int j = 0; j < featureMapNumC1; j++) {
				convolution(S2[j], kernelC3[i * 6 + j], C3[i]);
			}
			//add bias
			for (int x = 0; x < C3[i].length; x++) {
				for (int y = 0; y < C3[i].length; y++) {
					C3[i][x][y] += biasC3[i];
				}
			}


			//C3 to S4: mean pooling
			int sizeS = (int) ((double) sizeC / 2);
			S4[i] = new double[sizeS][sizeS];
			meanPooling(C3[i], S4[i], biasS4[i]);
		}

		//S4 to C5: convolution
		for (int i = 0; i < featureMapNumC5; i++) {
			double sum = 0d;
			for (int j = 0; j < featureMapNumC3; j++) {
				sum += convolution(S4[j], kernelC5[i * 16 + j]);
			}
			C5[i] = sum + biasC5[i];
		}


		//bp
		for (int i = 0; i < featureMapNumC5; i++) {
			bp.setInput(i, C5[i]);
		}

		bp.reInitDesire();
		for (int i = 0; i < outputNum; i++) {
			bp.setDesire(i, this.desire[i]);
		}


		bp.clearLayer();
		bp.forward();

	}

	//print layers info
	public void printLayer() {
		//kernel
		System.out.println();
		System.out.println("--------------------- kernelC1 ----------------------");
		for (int i = 0; i < kernelC1[0].length; i++) {
			for (int j = 0; j < kernelC1[0].length; j++) {
				System.out.print(kernelC1[0][i][j] + " ");
			}
			System.out.println();
		}

		//bias
		System.out.println();
		System.out.println("--------------------- biasC1 ----------------------");
		for (int i = 0; i < biasC1.length; i++) {
			System.out.print(biasC1[i] + " ");
		}


		//input
		System.out.println();
		System.out.println("--------------------- input ----------------------");
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[i].length; j++) {
				System.out.print(df.format(input[i][j]) + " ");
			}
			System.out.println();
		}


		//C1
		System.out.println();
		System.out.println("--------------------- C1 ----------------------");
		for (int i = 0; i < C1[0].length; i++) {
			for (int j = 0; j < C1[0].length; j++) {
				System.out.print(df.format(C1[0][i][j]) + " ");
			}
			System.out.println();
		}


		//S2
		System.out.println();
		System.out.println("--------------------- S2 ----------------------");
		for (int i = 0; i < S2[0].length; i++) {
			for (int j = 0; j < S2[0].length; j++) {
				System.out.print(df.format(S2[0][i][j]) + " ");
			}
			System.out.println();
		}

		//C3
		System.out.println();
		System.out.println("--------------------- C3 ----------------------");
		for (int i = 0; i < C3[0].length; i++) {
			for (int j = 0; j < C3[0].length; j++) {
				System.out.print(C3[0][i][j] + " ");
			}
			System.out.println();
		}

		//S4
		System.out.println();
		System.out.println("--------------------- S4 ----------------------");
		for (int i = 0; i < S4[0].length; i++) {
			for (int j = 0; j < S4[0].length; j++) {
				System.out.print(S4[0][i][j] + " ");
			}
			System.out.println();
		}

		System.out.println();
		System.out.println("--------------------- C5 ----------------------");
		for (int i = 0; i < featureMapNumC5; i++) {
			System.out.print(C5[i] + " ");
		}

	}


	//print sensitivity info
	public void printSensitivity() {
		//C5
		System.out.println();
		System.out.println("--------------------- C5 sen ----------------------");
		for (int i = 0; i < featureMapNumC5; i++) {
			System.out.print(sensitivityC5[i][0][0] + " ");
		}

		//S4
		System.out.println();
		System.out.println("--------------------- S4 sen ----------------------");
		for (int i = 0; i < sensitivityS4[0].length; i++) {
			for (int j = 0; j < sensitivityS4[0].length; j++) {
				System.out.print(sensitivityS4[0][i][j] + " ");
			}
			System.out.println();
		}
	}


	//calculate the error and store the sensitivities
	public void calculateError() {
		//BP backward
		bp.backward();

		//BP to C5
		for (int i = 0; i < featureMapNumC5; i++) {
			sensitivityC5[i][0][0] = bp.getInputError(i);
		}

		//C5 to S4:convolution
		for (int i = 0; i < featureMapNumC3; i++) {
			sensitivityS4[i] = new double[S4[0].length][S4[0].length];
			for (int j = 0; j < featureMapNumC5; j++) {
				double[][] tempSensitivity = enlargeMatrix(sensitivityC5[j]);
				double[][] rotKernel = rot180(kernelC5[i * 120 + j]);
				convolution(tempSensitivity, rotKernel, sensitivityS4[i]);
			}
		}

		//S4 to C3: back pooling
		for (int i = 0; i < featureMapNumC3; i++) {
			sensitivityC3[i] = backPooling(sensitivityS4[i]);
		}


		//C3 to S2: convolution
		for (int i = 0; i < featureMapNumC1; i++) {
			sensitivityS2[i] = new double[S2[0].length][S2[0].length];
			for (int j = 0; j < featureMapNumC3; j++) {
				double[][] tempSensitivity = enlargeMatrix(sensitivityC3[j]);
				double[][] rotKernel = rot180(kernelC3[i * 16 + j]);
				convolution(tempSensitivity, rotKernel, sensitivityS2[i]);
			}
		}


		//S2 to C1: back pooling
		for (int i = 0; i < featureMapNumC1; i++) {
			sensitivityC1[i] = backPooling(sensitivityS2[i]);
		}


		//C1 to input: convolution
		for (int i = 0; i < featureMapNumC1; i++) {
			sensitivityInput = new double[inputNum][inputNum];
			double[][] tempSensitivity = enlargeMatrix(sensitivityC1[i]);
			double[][] rotKernel = rot180(kernelC1[i]);
			convolution(tempSensitivity, rotKernel, sensitivityInput);
		}

	}


	//backward: adjust bias and kernels
	public void backward() {

		//C5
		for (int i = 0; i < featureMapNumC5; i++) {
			//bias
			double deltaBias = 0d;
			deltaBias = sensitivityC5[i][0][0];
			biasC5[i] += deltaBias * learningRate;

			//kernel
			for (int j = 0; j < featureMapNumC3; j++) {
				double[][] deltaKernel = new double[kernelSize][kernelSize];
				convolution(S4[j], sensitivityC5[i], deltaKernel);
				for (int x = 0; x < kernelSize; x++) {
					for (int y = 0; y < kernelSize; y++) {
						kernelC5[i * 16 + j][x][y] += deltaKernel[x][y] * learningRate;
					}
				}

			}
		}

		//S4 bias
		for (int i = 0; i < featureMapNumC3; i++) {
			double deltaBias = 0d;
			for (int x = 0; x < sensitivityS4[i].length; x++) {
				for (int y = 0; y < sensitivityS4[i][x].length; y++) {
					deltaBias += sensitivityS4[i][x][y];
				}
			}
			biasS4[i] += deltaBias * learningRate;
		}


		//C3
		for (int i = 0; i < featureMapNumC3; i++) {
			//bias
			double deltaBias = 0d;
			for (int x = 0; x < sensitivityC3[i].length; x++) {
				for (int y = 0; y < sensitivityC3[i][x].length; y++) {
					deltaBias += sensitivityC3[i][x][y];
				}
			}
			biasC3[i] += deltaBias * learningRate / 25;

			//kernel
			for (int j = 0; j < featureMapNumC1; j++) {
				double[][] deltaKernel = new double[kernelSize][kernelSize];
				convolution(S2[j], sensitivityC3[i], deltaKernel);
				for (int x = 0; x < kernelSize; x++) {
					for (int y = 0; y < kernelSize; y++) {
						kernelC3[i * 6 + j][x][y] += deltaKernel[x][y] * learningRate / 25;
					}
				}

			}
		}


		//S2 bias
		for (int i = 0; i < featureMapNumC1; i++) {
			double deltaBias = 0d;
			for (int x = 0; x < sensitivityS2[i].length; x++) {
				for (int y = 0; y < sensitivityS2[i][x].length; y++) {
					deltaBias += sensitivityS2[i][x][y];
				}
			}
			biasS2[i] += deltaBias * learningRate / 25;
		}


		//C1
		for (int i = 0; i < featureMapNumC1; i++) {
			//bias
			double deltaBias = 0d;
			for (int x = 0; x < sensitivityC1[i].length; x++) {
				for (int y = 0; y < sensitivityC1[i][x].length; y++) {
					deltaBias += sensitivityC1[i][x][y] / 25;
				}
			}

			//kernel
			double[][] deltaKernel = new double[kernelSize][kernelSize];
			convolution(input, sensitivityC1[i], deltaKernel);
			for (int x = 0; x < kernelSize; x++) {
				for (int y = 0; y < kernelSize; y++) {
					kernelC1[i][x][y] += deltaKernel[x][y] * learningRate / 25;
				}
			}

		}


		reInitMatrix(input);
		reInitMatrix(C1);
		reInitMatrix(S2);
		reInitMatrix(C3);
		reInitMatrix(S4);
		reInitMatrix(C5);

		reInitMatrix(sensitivityInput);
		reInitMatrix(sensitivityC1);
		reInitMatrix(sensitivityS2);
		reInitMatrix(sensitivityC3);
		reInitMatrix(sensitivityS4);
		reInitMatrix(sensitivityC5);


	}


	//180бу rotation
	public double[][] rot180(double[][] kernel) {
		int size = kernel.length;
		double[][] tempKernel = new double[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				tempKernel[i][j] = kernel[size - 1 - i][size - 1 - j];
			}
		}
		return tempKernel;
	}


	//to enlarge a matrix by adding 4 rows of 0
	public double[][] enlargeMatrix(double[][] matrix) {
		int size = matrix.length + 8;
		double[][] temp = new double[size][size];

		for (int i = 0; i < size; i++) {
			temp[i][0] = 0;
			temp[i][1] = 0;
			temp[i][2] = 0;
			temp[i][3] = 0;
			temp[i][size - 1] = 0;
			temp[i][size - 2] = 0;
			temp[i][size - 3] = 0;
			temp[i][size - 4] = 0;
		}

		for (int j = 0; j < size; j++) {
			temp[0][j] = 0;
			temp[1][j] = 0;
			temp[2][j] = 0;
			temp[3][j] = 0;
			temp[size - 1][j] = 0;
			temp[size - 2][j] = 0;
			temp[size - 3][j] = 0;
			temp[size - 4][j] = 0;
		}

		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				temp[i + 4][j + 4] = matrix[i][j];
			}
		}

		return temp;
	}


	//up-sampling
	public double[][] backPooling(double[][] matrix) {
		int size = matrix.length;
		double[][] temp = new double[size * 2][size * 2];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				double mean = matrix[i][j] / 4;
				temp[i * 2][j * 2] = mean;
				temp[i * 2][j * 2 + 1] = mean;
				temp[i * 2 + 1][j * 2] = mean;
				temp[i * 2 + 1][j * 2 + 1] = mean;
			}
		}
		return temp;
	}


	//ReLU function
	public double ReLU(double x) {
		double result = 0.0d;
		if (x > 0) {
			result = x;
		}

		return result;
	}


	//to get the desire result
	public int getDesire() {
		int index = 0;
		for (int i = 0; i < outputNum; i++) {
			if (desire[i] == 1) {
				index = i;
			}
		}
		return index;
	}


}
