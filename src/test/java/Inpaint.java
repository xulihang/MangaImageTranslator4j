import ai.onnxruntime.OrtException;
import com.xulihang.AOTInpainter;
import com.xulihang.OCRCTC;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;

public class Inpaint {
    public static void main(String[] args) throws Exception {
        // 加载OpenCV本地库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String modelPath = "model.onnx";
        String imagePath = "image.jpg";
        String maskPath = "mask.png";
        String outputPath = "repaired_java.png";

        AOTInpainter inpainter = new AOTInpainter(modelPath, false);
        inpainter.inpaint(imagePath, maskPath, outputPath);

    }
}
