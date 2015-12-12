import java.io.*;

/**
 * Created by W on 2015/10/24.
 */
public class PartB {
	private static long serialVersionUID = 1;
	private CNN cnn;
	public String networkSettingFile;  //to store configuration


	PartB(String networkSettingFile) {
		/*constructor:
		* CNN(int inputNum,int outputNum,int featureMapC1,int featureMapC3,int featureMapC5, int kernelSize,int stride,double learningRate,double RegularizationRate)
		*/
		cnn = new CNN(32, 10, 6, 16, 120, 5, 1, 0.005, 0.05);
		this.networkSettingFile = networkSettingFile;

	}

	//load the input data from training sets
	public void loadInputTrainingData(String fileName) {
		File file;
		try {
			String encoding = "utf-8";
			file = new File(fileName);
			if (file.isFile() && file.exists()) {
				InputStreamReader reader = new InputStreamReader(new FileInputStream(file), encoding);
				BufferedReader bufferedReader = new BufferedReader(reader);
				String line = null;
				int lineNum = 0;
				while ((line = bufferedReader.readLine()) != null && lineNum < 28) {
					String[] data = line.split(" ");
					for (int i = 0; i < 28; i++) {
						double number = Double.parseDouble(data[i]);
//						cnn.setInput(lineNum + 2, i + 2, (number - 128) / 128);
						cnn.setInput(lineNum + 2, i + 2, number / 2);
					}
					lineNum++;
				}
				//load the desire result
				if (lineNum == 28) {
					int index = Integer.parseInt(line);
					cnn.reInitDesire();
					cnn.setDesire(index, 1);
				}
				reader.close();

				//add 0
				for (int i = 0; i < 32; i++) {
					cnn.setInput(i, 0, 0);
					cnn.setInput(i, 1, 0);
					cnn.setInput(i, 30, 0);
					cnn.setInput(i, 31, 0);
				}

				for (int j = 0; j < 32; j++) {
					cnn.setInput(0, j, 0);
					cnn.setInput(1, j, 0);
					cnn.setInput(30, j, 0);
					cnn.setInput(31, j, 0);
				}

			} else {
				System.out.println("File not exist");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


	//load the input data from training sets
	public void loadInputTestingData(String fileName) {
		File file;
		try {
			String encoding = "utf-8";
			file = new File(fileName);
			if (file.isFile() && file.exists()) {
				InputStreamReader reader = new InputStreamReader(new FileInputStream(file), encoding);
				BufferedReader bufferedReader = new BufferedReader(reader);
				String line = null;
				int lineNum = 0;
				while ((line = bufferedReader.readLine()) != null && lineNum < 28) {
					String[] data = line.split(" ");
					for (int i = 0; i < 28; i++) {
						double number = Double.parseDouble(data[i]);
//						cnn.setInput(lineNum + 2, i + 2, (number - 128) / 128);
						cnn.setInput(lineNum + 2, i + 2, number);
					}
					lineNum++;
				}
				reader.close();

				//add 0
				for (int i = 0; i < 32; i++) {
					cnn.setInput(i, 0, 0);
					cnn.setInput(i, 1, 0);
					cnn.setInput(i, 30, 0);
					cnn.setInput(i, 31, 0);
				}

				for (int j = 0; j < 32; j++) {
					cnn.setInput(0, j, 0);
					cnn.setInput(1, j, 0);
					cnn.setInput(30, j, 0);
					cnn.setInput(31, j, 0);
				}

			} else {
				System.out.println("File not exist");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


	//training
	public void train(String filePath, int trainingTime, int sampleNum) {
		for (int i = 1; i <= trainingTime; i++) {
			switch (i) {
				case 1:
					cnn.setLearningRate(0.005);
					break;
				case 4:
					cnn.setLearningRate(0.003);
					break;
				case 6:
					cnn.setLearningRate(0.001);
					break;
				case 12:
					cnn.setLearningRate(0.0005);
					break;
				case 16:
					cnn.setLearningRate(0.0001);
					break;
//				case 18:
//					cnn.setLearningRate(0.00005);
//					break;
			}
			double accurateNum = 0;
			for (int j = 1; j <= sampleNum; j++) {
				loadInputTrainingData(filePath + j + ".txt");
				cnn.forward();
				if (cnn.bp.isOutputCorrect()) {
					accurateNum++;
				}
				System.out.println("[training times] " + (i - 1) + "*" + sampleNum + " + " + j + ",  [accurate number] " + accurateNum + ",   " + "[accuracy] " + ((double) accurateNum / (double) j));
				cnn.calculateError();
				cnn.backward();

			}

		}

	}



	//test the testing sets and calculate accuracy
	public void test(String filePath, int start, int sampleNum) {
		double accurateNum = 0;
		for (int i = 1; i <= sampleNum; i++) {
			int fileNum = i + start;
			loadInputTrainingData(filePath + fileNum + ".txt");
			cnn.forward();
			if (cnn.bp.isOutputCorrect()) {
				accurateNum++;
			}
			System.out.println("[testing times] " + i + ",  [accurate number] " + accurateNum + ",   " + "[accuracy] " + ((double) accurateNum / (double) i));
		}
	}



	//test the testing sets
	public void test(String filePath, int sampleNum) {
		File file = new File("13302010104.txt");
		FileWriter fileWriter = null;
		try {
			fileWriter = new FileWriter(file);
			for (int i = 1; i <= sampleNum; i++) {
				loadInputTestingData(filePath + i + ".txt");
				cnn.forward();
				String result = cnn.bp.getOutput() + "";
				System.out.println(result);
				fileWriter.write(result + "\r\n");
			}
			fileWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	//serialize the CNN
	private void serializeCNN(CNN cnn) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(networkSettingFile));
			out.writeObject(cnn); //
			System.out.println("[serialization] data has been stored.");
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}


	//deserialize the CNN
	private CNN deserializeCNN() {
		CNN cnn = null;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(networkSettingFile));
			cnn = (CNN) in.readObject();
			System.out.println("[deserialization] data has been read.");
			in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return cnn;
	}



	public static void main(String[] args) {
		PartB lab = new PartB("cnn_test4");

//		//training and testing
//		lab.train("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/", 30, 3000);
//		lab.serializeCNN(lab.cnn);
//		lab.test("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/", 1, 200);



		//testing
		lab.cnn=lab.deserializeCNN();
//		lab.test("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/",1, 200);
		lab.test("F:/test799/",799);

	}
}
