import java.io.*;

/**
 * Created by W on 2015/10/3.
 */
public class PartA implements Serializable {
	private static long serialVersionUID = 1;

	private BP bp;  //BP network

	public String networkSettingFile;  //to store configuration
	public int sampleNum;  //training sample number
	public double learningRate;  //learning rate

	public int inputNum; //input layer number
	public int hiddenNum; //hidden layer number
	public int outputNum; //output layer number


	public PartA(String settingFileName) {
		networkSettingFile = settingFileName;
		bp = deserializeBP();
	}

	public PartA(int inputNum, int hiddenNum, int outputNum, double learningRate, int sampleNum, String settingFileName) {
		bp = new BP(inputNum, hiddenNum, outputNum, learningRate);
		networkSettingFile = settingFileName;
		this.sampleNum = sampleNum;

		this.inputNum = inputNum;
		this.hiddenNum = hiddenNum;
		this.outputNum = outputNum;
		this.learningRate = learningRate;
	}


	//load the input data from training sets
	public void loadInputTrainingData(String fileName) {
		File file;
		try {
			String encoding = "utf-8";
//			System.out.println("training :"+fileName);
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
						//bp.setInput(i + lineNum * 28, (number - 128) / 128);
						bp.setInput(i + lineNum * 28, number / 255);
//						input[i + lineNum * 28] = (number - 128) / 128;
					}
					lineNum++;
				}
				//load the desire result
				if (lineNum == 28) {
					int index = Integer.parseInt(line);
					bp.reInitDesire();
					bp.setDesire(index, 1);
				}
				reader.close();
			} else {
				System.out.println("File not exist");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


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
						bp.setInput(i + lineNum * 28, (number - 128) / 128);
//						bp.setInput(i + lineNum * 28, number / 255);
//						input[i + lineNum * 28] = (number - 128) / 128;
					}
					lineNum++;
				}
				reader.close();
			} else {
				System.out.println("File not exist");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


	public void train() {
		double accurateNum = 0;
		for (int i = 1; i <= sampleNum; i++) {
			loadInputTrainingData("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/" + i + ".txt");
			bp.clearLayer();
			bp.forward();
			if (bp.isOutputCorrect()) {
				accurateNum++;
			}
			System.out.println("[training times] " + i + ",  [accurate number] " + accurateNum + ",   " + "[accuracy] " + ((double) accurateNum / (double) i));
			bp.backward();
		}
	}


	//test the default testing set
	public void test() {
		int accurateNum = 0;
//		int start=this.sampleNum;
		int start = 2501;
		int sampleNum = 500;
		for (int i = start; i < start + sampleNum; i++) {
			loadInputTrainingData("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/" + i + ".txt");
			bp.forward();
			if (bp.isOutputCorrect()) {
				accurateNum++;
			}
			System.out.println("[testing] " + i + ",  [accurate number] " + accurateNum + ",   " + "[accuracy] " + ((double) accurateNum / (double) (i - start + 1)));
		}
	}


	//test a specific test set
	public void test(String path, int sampleNum) {
		File file = new File("13302010104.txt");
		FileWriter fileWriter = null;
		try {
			fileWriter = new FileWriter(file);
			for (int i = 1; i <= sampleNum; i++) {
				loadInputTestingData(path + i + ".txt");
				bp.forward();
				String result = bp.getOutput() + "";
				System.out.println(result);
				fileWriter.write(result + "\r\n");
			}
			fileWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}


	//serialize the BP Network
	private void serializeBP(BP bp) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(networkSettingFile));
			out.writeObject(bp); //
			System.out.println("[serialization] data has been stored.");
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}


	//deserialize the BP Network
	private BP deserializeBP() {
		BP bp = null;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(networkSettingFile));
			bp = (BP) in.readObject();
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

		return bp;
	}


	public static void main(String[] args) {

		//training and testing
		PartA lab = new PartA(784, 100, 10, 0.05, 2500, "test0");
		int learningTime = 10;
		for (int i = 0; i < learningTime; i++) {
			lab.train();
		}
//			lab.serializeBP(lab.bp);
//
		lab.test();
//			lab.test("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/",10);


//			//testing
//			PartA lab=new PartA("test0");
//			lab.test();
//			lab.test("F:/school/IntelligentSystem/LAB/Lab1/dataset/newTestSet/",10);

	}

}

