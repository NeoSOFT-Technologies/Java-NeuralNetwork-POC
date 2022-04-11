package djl_import;
import java.nio.file.*;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.*;
import ai.djl.modality.cv.translator.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;

public class App {
	public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {	
		
		Pipeline pipeline = new Pipeline();
		pipeline.add(new Resize(640, 640));
		pipeline.add(new ToTensor());
		
		Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().setPipeline(pipeline).optSynsetArtifactName("synset.txt")
				.optThreshold(0.5f)
				.build();
		
		Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelPath(Paths.get("C:\\Users\\A\\Desktop\\YOLO5_train\\Weights\\pytorchModel\\best.torchscript"))
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

		ZooModel<Image, DetectedObjects> model = criteria.loadModel();
		
		var img = ImageFactory.getInstance().fromFile(Paths.get("C:\\Users\\A\\Desktop\\YOLO5_train\\my_new_images\\2548.jpg"));
		img.getWrappedImage();
		
		Predictor<Image, DetectedObjects> predictor = model.newPredictor();
		DetectedObjects Detected_dents = predictor.predict(img);
		System.out.println(Detected_dents);
		
		
		DrawBoundingBoxes.draw(Detected_dents, "C:\\Users\\A\\Desktop\\YOLO5_train\\output_from_djl\\resized\\2548R.jpg");

	}
}
