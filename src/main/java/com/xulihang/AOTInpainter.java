package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class AOTInpainter {

    private final OrtEnvironment env;
    private final OrtSession session;

    public AOTInpainter(String modelPath, boolean useCUDA) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        if (useCUDA) {
            options.addCUDA(0);
            System.out.println("✅ 使用 GPU 推理");
        } else {
            System.out.println("✅ 使用 CPU 推理");
        }

        session = env.createSession(modelPath, options);
    }

    private static int alignToMultiple(int x, int base) {
        return ((x + base - 1) / base) * base;
    }

    private static Mat padToMultiple(Mat img, int base) {
        int h = img.rows();
        int w = img.cols();
        int newH = alignToMultiple(h, base);
        int newW = alignToMultiple(w, base);
        int padH = newH - h;
        int padW = newW - w;

        Mat padded = new Mat();
        Core.copyMakeBorder(img, padded, 0, padH, 0, padW, Core.BORDER_REFLECT);
        return padded;
    }

    private static float[] matToCHWFloat(Mat img) {
        int h = img.rows();
        int w = img.cols();
        int c = img.channels();
        float[] data = new float[c * h * w];
        byte[] pixels = new byte[c * h * w];
        img.get(0, 0, pixels);
        int idx = 0;
        for (int ci = 0; ci < c; ci++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int bi = (y * w + x) * c + ci;
                    int val = pixels[bi] & 0xFF;
                    data[idx++] = val / 255.0f;
                }
            }
        }
        return data;
    }

    private static float[] maskToFloat(Mat mask) {
        int h = mask.rows();
        int w = mask.cols();
        float[] data = new float[h * w];
        byte[] pixels = new byte[h * w];
        mask.get(0, 0, pixels);
        for (int i = 0; i < pixels.length; i++) {
            data[i] = (pixels[i] & 0xFF) > 127 ? 1.0f : 0.0f;
        }
        return data;
    }

    public Mat inpaint(String imagePath, String maskPath, String outputPath) throws Exception {
        // 1️⃣ 读取图像和mask
        Mat image = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_COLOR);
        Mat mask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE);
        if (image.empty() || mask.empty()) {
            throw new RuntimeException("无法读取图像或掩码。");
        }

        int origH = image.rows();
        int origW = image.cols();

        // 2️⃣ 对齐尺寸到8倍
        image = padToMultiple(image, 8);
        mask = padToMultiple(mask, 8);

        // 清空 mask 区域
        for (int y = 0; y < mask.rows(); y++) {
            for (int x = 0; x < mask.cols(); x++) {
                if (mask.get(y, x)[0] > 127) {
                    image.put(y, x, new double[]{0, 0, 0});
                }
            }
        }

        // 3️⃣ 转为Tensor输入
        int H = image.rows();
        int W = image.cols();
        float[] imageData = matToCHWFloat(image);
        float[] maskData = maskToFloat(mask);

        OnnxTensor imageTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData),
                new long[]{1, 3, H, W});
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(maskData),
                new long[]{1, 1, H, W});

        // 4️⃣ 推理
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("image", imageTensor);
        inputs.put("mask", maskTensor);

        OrtSession.Result result = session.run(inputs);
        OnnxValue outValue = result.get(0);
        float[][][][] output = (float[][][][]) outValue.getValue(); // [1,3,H,W]

        // 5️⃣ 后处理
        Mat outputImg = new Mat(H, W, CvType.CV_8UC3);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                double[] pix = new double[3];
                for (int c = 0; c < 3; c++) {
                    float val = output[0][c][y][x];
                    pix[c] = Math.max(0, Math.min(255, val * 255.0f));
                }
                outputImg.put(y, x, pix);
            }
        }

        // 6️⃣ 裁剪回原始大小
        Mat cropped = new Mat(outputImg, new Rect(0, 0, origW, origH));
        Imgcodecs.imwrite(outputPath, cropped);
        System.out.println("✅ 修复完成，已保存到：" + outputPath);

        return cropped;
    }
}