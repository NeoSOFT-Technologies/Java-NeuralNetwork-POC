package djl_import;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import ai.djl.modality.cv.output.DetectedObjects;

import java.text.DecimalFormat;

import org.json.*;

public class DrawBoundingBoxes {
	public static void draw(DetectedObjects Detected_dents,String ImagePath) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat matrix = Imgcodecs.imread(ImagePath);
		System.out.println(Detected_dents.item(0).getClassName());
		System.out.println(Detected_dents.item(0).getProbability());
		String res = Detected_dents.toJson();
		
		System.out.println(res);
		String jsonString = res;
		JSONArray jsonArr = new JSONArray(jsonString);
		
		for(int i=0;i<jsonArr.length();i++) {
			JSONObject  jsonObj = jsonArr.getJSONObject(i);
			
			String detectedClassName = jsonObj.getString("className");
			double probability = jsonObj.getDouble("probability");
			DecimalFormat df = new DecimalFormat("0.00");	
			String textToAdd = detectedClassName +"("+ df.format(probability)+")";
			
			JSONObject firstPoint = jsonObj.getJSONObject("boundingBox").getJSONArray("corners").getJSONObject(0);
			double x1 = firstPoint.getDouble("x");
			double y1 = firstPoint.getDouble("y");
			
			JSONObject secondPoint = jsonObj.getJSONObject("boundingBox").getJSONArray("corners").getJSONObject(2);
			double x2 = secondPoint.getDouble("x");
			double y2 = secondPoint.getDouble("y");			
			
			Imgproc.rectangle(matrix, // Matrix obj of the image
					new Point(x1, y1), // p1
					new Point(x2, y2), // p2
					new Scalar(0, 0, 255), // Scalar object for color
					2 // Thickness of the line
			);

			Imgproc.putText(matrix, textToAdd, new Point(x1, y1-5), 4, 0.6, new Scalar(0, 0, 255), 1);
		}	
		         
//		Imgcodecs.imwrite("../djl_import/res.jpg", matrix); // to save an image

		// display Image
		HighGui.imshow("Result", matrix);
		// Waiting for a key event to delay
		HighGui.waitKey();

	}
}
