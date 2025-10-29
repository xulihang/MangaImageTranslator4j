import ai.onnxruntime.OrtException;
import com.xulihang.AOTInpainter;
import com.xulihang.OCRCTC;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class Inpaint {
    public static void main(String[] args) throws Exception {
        // 加载OpenCV本地库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String modelPath = "model.onnx";
        String imagePath = "origin.jpg";
        String maskPath = "origin.jpg-mask.png";
        String outputPath = "repaired_java.png";
        Mat image = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_COLOR);
        Mat mask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat thresh = new Mat();
        Imgproc.threshold(mask,thresh,200,255,Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        AOTInpainter inpainter = new AOTInpainter(modelPath, false);
        Mat out = inpainter.inpaint(image, thresh);
        Imgcodecs.imwrite(outputPath, out);
    }
}
