package com.xulihang;

import ai.onnxruntime.OrtException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws OrtException, IOException {
        // 加载OpenCV本地库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);


        // 初始化ONNX OCR模型
        OCRCTC onnxOcr = new OCRCTC(
                "ocr_ctc.onnx",
                "alphabet-all-v5.txt"
        );

        Mat img = Imgcodecs.imread("test.jpg");

        // 进行OCR推理
        List<OCRCTC.Result> results = onnxOcr.infer(img, true);
        System.out.println(results);




    }
}