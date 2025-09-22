package com.xulihang;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import ai.onnxruntime.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class OCRCTC {
    private List<String> dictionary;
    private int blank;
    private int dictSize;
    private OrtSession session;
    private String inputName;
    private List<String> outputNames;
    private OrtEnvironment env;

    public OCRCTC(String onnxModelPath, String dictionaryPath, int blank) throws OrtException, IOException {
        // 加载字典
        dictionary = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(dictionaryPath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                dictionary.add(line.trim());
            }
        }

        this.blank = blank;
        this.dictSize = dictionary.size();

        // 创建ONNX Runtime环境
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 加载ONNX模型
        session = env.createSession(onnxModelPath, sessionOptions);

        // 获取输入和输出名称
        inputName = session.getInputNames().iterator().next();
        outputNames = new ArrayList<>();
        for (String name : session.getOutputNames()) {
            outputNames.add(name);
        }

        System.out.println("成功加载ONNX模型: " + onnxModelPath);
        System.out.println("字典大小: " + dictSize);
        System.out.println("ONNX输入名称: " + inputName);
        System.out.println("ONNX输出名称: " + outputNames);
    }

    public OCRCTC(String onnxModelPath, String dictionaryPath) throws OrtException, IOException {
        this(onnxModelPath, dictionaryPath, 0);
    }

    public FloatBuffer preprocess(Mat image) {
        // 确保输入图像尺寸正确
        if (image.rows() != 48) {
            // 保持宽高比缩放
            int h = image.rows();
            int w = image.cols();
            float ratio = 48.0f / h;
            int newW = (int) (w * ratio);
            Imgproc.resize(image, image, new Size(newW, 48), 0, 0, Imgproc.INTER_LINEAR);
        }

        // 转换为RGB格式
        if (image.channels() == 1) {
            Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2RGB);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGRA2RGB);
        }

        // 归一化处理
        Mat floatImage = new Mat();
        image.convertTo(floatImage, CvType.CV_32F);
        Core.subtract(floatImage, new Scalar(127.5, 127.5, 127.5), floatImage);
        Core.divide(floatImage, new Scalar(127.5, 127.5, 127.5), floatImage);

        // 获取图像尺寸
        int height = floatImage.rows();
        int width = floatImage.cols();
        int channels = floatImage.channels();

        // 创建FloatBuffer并填充数据 [C, H, W] 顺序
        FloatBuffer buffer = FloatBuffer.allocate(channels * height * width);

        // 将OpenCV Mat数据转换为CHW格式
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float[] pixel = new float[channels];
                    floatImage.get(h, w, pixel);
                    buffer.put(pixel[c]);
                }
            }
        }
        buffer.rewind();
        floatImage.release();
        return buffer;
    }

    public OCRResult infer(Mat image) throws OrtException {

        // 预处理图像
        FloatBuffer preprocessedBuffer = preprocess(image);

        // 获取图像尺寸信息
        int channels = 3; // RGB图像
        int height = image.rows();
        int width = image.cols();

        // 创建输入形状 [batch, channels, height, width]
        long[] shape = {1, channels, height, width};

        // 创建ONNX输入张量
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(
                env,
                preprocessedBuffer,
                shape
        )) {

            // 执行ONNX推理
            try (OrtSession.Result outputs = session.run(Collections.singletonMap(inputName, inputTensor))) {

                // 安全地处理输出 - 根据实际维度进行处理
                float[][][] charLogits = null;
                float[][][] colorValues = null;

                // 尝试不同的维度
                for (String outputName : outputNames) {
                    OnnxValue value = outputs.get(outputName).get();
                    if (value instanceof OnnxTensor) {
                        OnnxTensor tensor = (OnnxTensor) value;
                        long[] tensorShape = tensor.getInfo().getShape();

                        System.out.println("Output " + outputName + " shape: " + java.util.Arrays.toString(tensorShape));

                        // 根据形状确定是3维还是4维
                        if (tensorShape.length == 3) {
                            if (charLogits == null) {
                                charLogits = (float[][][]) tensor.getValue();
                            } else if (colorValues == null) {
                                colorValues = (float[][][]) tensor.getValue();
                            }
                        } else if (tensorShape.length == 4) {
                            float[][][][] temp4D = (float[][][][]) tensor.getValue();
                            // 将4D转换为3D（去掉不必要的维度）
                            if (temp4D.length == 1) {
                                if (charLogits == null) {
                                    charLogits = temp4D[0];
                                } else if (colorValues == null) {
                                    colorValues = temp4D[0];
                                }
                            }
                        }
                    }
                }

                if (charLogits != null) {
                    return decodeCtcTop1(charLogits, colorValues);
                } else {
                    System.err.println("未能获取到有效的字符logits输出");
                    return new OCRResult("", 0.0f, new ArrayList<>());
                }
            }
        } catch (Exception e) {
            System.err.println("推理过程中出错: " + e.getMessage());
            e.printStackTrace();
            return new OCRResult("", 0.0f, new ArrayList<>());
        }
    }

    private OCRResult decodeCtcTop1(float[][][] predCharLogits, float[][][] predColorValues) {
        List<CharResult> charResults = new ArrayList<>();
        int batchSize = predCharLogits.length;
        int timeSteps = predCharLogits[0].length;
        int numClasses = predCharLogits[0][0].length;

        System.out.println("Char logits shape: [" + batchSize + ", " + timeSteps + ", " + numClasses + "]");

        if (predColorValues != null) {
            System.out.println("Color values shape: [" + predColorValues.length + ", " +
                    (predColorValues.length > 0 ? predColorValues[0].length : 0) + ", " +
                    (predColorValues.length > 0 && predColorValues[0].length > 0 ? predColorValues[0][0].length : 0) + "]");
        }

        // 处理批次中的每个样本
        for (int b = 0; b < batchSize; b++) {
            int lastCh = blank;

            for (int t = 0; t < timeSteps; t++) {
                // 找到最大概率的字符索引
                int predCh = blank;
                float maxProb = Float.NEGATIVE_INFINITY;

                for (int c = 0; c < numClasses; c++) {
                    if (predCharLogits[b][t][c] > maxProb) {
                        maxProb = predCharLogits[b][t][c];
                        predCh = c;
                    }
                }

                // CTC解码：跳过空白符和连续重复字符
                if (predCh != lastCh && predCh != blank) {
                    // 计算log probability
                    float sumExp = 0;
                    for (int c = 0; c < numClasses; c++) {
                        sumExp += (float) Math.exp(predCharLogits[b][t][c]);
                    }
                    float logProb = predCharLogits[b][t][predCh] - (float) Math.log(sumExp);

                    // 提取颜色值（如果有）
                    float fr = 0, fg = 0, fb = 0, br = 0, bg = 0, bb = 0;
                    if (predColorValues != null && predColorValues.length > b && predColorValues[b].length > t) {
                        float[] colorArray = predColorValues[b][t];
                        if (colorArray.length >= 6) {
                            fr = colorArray[0]; fg = colorArray[1]; fb = colorArray[2];
                            br = colorArray[3]; bg = colorArray[4]; bb = colorArray[5];

                            // 确保颜色值在0-1范围内（如果是sigmoid输出）
                            fr = Math.max(0, Math.min(1, fr));
                            fg = Math.max(0, Math.min(1, fg));
                            fb = Math.max(0, Math.min(1, fb));
                            br = Math.max(0, Math.min(1, br));
                            bg = Math.max(0, Math.min(1, bg));
                            bb = Math.max(0, Math.min(1, bb));
                        }
                    }

                    charResults.add(new CharResult(
                            dictionary.get(predCh), logProb,
                            fr, fg, fb,
                            br, bg, bb
                    ));
                }

                lastCh = predCh;
            }
        }

        // 将预测结果转换为文本和颜色信息
        StringBuilder text = new StringBuilder();
        float prob = 0;
        for (CharResult charResult: charResults) {

            float totalLogprob = 0;

            String ch = charResult.character;
            if (ch.equals("<SP>")) {
                ch = " ";
            }
            text.append(ch);
            totalLogprob += charResult.logProb;


            // 计算概率
            if (!charResults.isEmpty()) {
                prob = (float) Math.exp(totalLogprob / charResults.size());
            }

        }
        OCRResult result = new OCRResult(text.toString(),prob,charResults);
        return result;
    }

    // 内部类用于存储字符结果
    // 简化的结果类
    public static class OCRResult {
        public String text;
        public float confidence;
        public List<CharResult> chars; // 每个字符的详细信息

        public OCRResult(String text, float confidence, List<CharResult> chars) {
            this.text = text;
            this.confidence = confidence;
            this.chars = chars;
        }
    }

    // 字符信息类
    public static class CharResult {
        String character;
        float logProb;
        float fr, fg, fb; // 前景色RGB (0-1范围)
        float br, bg, bb; // 背景色RGB (0-1范围)

        CharResult(String character, float logProb, float fr, float fg, float fb, float br, float bg, float bb) {
            this.character = character;
            this.logProb = logProb;
            this.fr = fr;
            this.fg = fg;
            this.fb = fb;
            this.br = br;
            this.bg = bg;
            this.bb = bb;
        }

        // 将0-1范围的浮点数转换为0-255的整数RGB值
        public int[] getForegroundRGB() {
            return new int[]{
                    (int) (fr * 255),
                    (int) (fg * 255),
                    (int) (fb * 255)
            };
        }

        public int[] getBackgroundRGB() {
            return new int[]{
                    (int) (br * 255),
                    (int) (bg * 255),
                    (int) (bb * 255)
            };
        }

        // 获取十六进制颜色代码
        public String getForegroundHex() {
            int[] rgb = getForegroundRGB();
            return String.format("#%02X%02X%02X", rgb[0], rgb[1], rgb[2]);
        }

        public String getBackgroundHex() {
            int[] rgb = getBackgroundRGB();
            return String.format("#%02X%02X%02X", rgb[0], rgb[1], rgb[2]);
        }
    }

}