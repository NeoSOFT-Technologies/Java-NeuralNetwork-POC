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
		
		String imgPath = "../djl_import/src/main/resources/sample_images/2548R.jpg";
		String modelPath = "../djl_import/src/main/resources/best.torchscript";
			
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
                        .optModelPath(Paths.get(modelPath))
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

		ZooModel<Image, DetectedObjects> model = criteria.loadModel();
		
		var img = ImageFactory.getInstance().fromFile(Paths.get(imgPath));
		img.getWrappedImage();
		
		
		Predictor<Image, DetectedObjects> predictor = model.newPredictor();
		DetectedObjects Detected_dents = predictor.predict(img);
		System.out.println(Detected_dents);
		
		DrawBoundingBoxes.draw(Detected_dents, imgPath);

	}
}
