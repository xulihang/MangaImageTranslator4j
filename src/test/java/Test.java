import com.xulihang.OCRCTC;
import ai.onnxruntime.OrtException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.util.List;

public class Test {
    public static void main(String[] args) throws OrtException, IOException {
        // 加载OpenCV本地库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);


        // 初始化ONNX OCR模型
        OCRCTC onnxOcr = new OCRCTC(
                "ocr_ctc_fp16.onnx",
                "alphabet-all-v5.txt"
        );

        Mat img = Imgcodecs.imread("clipped.jpg");
        for (int i = 0; i < 100; i++) {
            long startTime = System.currentTimeMillis();
            OCRCTC.OCRResult result = onnxOcr.infer(img);
            for (var charResult:result.chars) {
                System.out.println(charResult.character);
                System.out.println(charResult.fr);
                System.out.println(charResult.fg);
                System.out.println(charResult.fb);
            }
            System.out.println(result.text);
            long endTime = System.currentTimeMillis();
            System.out.println((endTime - startTime) + "ms");
        }
        // 进行OCR推理

    }
}