import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from ultralytics import YOLO
from PIL import Image
from urllib.request import urlretrieve
import logging

class CarVsTruck:  #class that has several functions within it 
    def __init__(self, model_path): # Function For the model path #Constructor that will load the model
        logging.info("CarVsTruck class initialized")
        self.model = YOLO(model_path) # YOLO: loads model structure and weights from .pt file
        logging.info("YOLO Model is loaded!")

    def predict(self, image_source):#function for loading the image after installing it and converting it to an array and it will give us the outcome.  
        # YOLOv8 handles the  source (URL or path) directly
        results = self.model(image_source)
        
        for r in results:
            if len(r.boxes) > 0:
                # FIX: Access the first element of the classes array
                top_class_index = int(r.boxes.cls[0])
                predicted_class = r.names[top_class_index]
                return predicted_class
            
        return "No object detected"

    def download_url(self, url, filename):
        urlretrieve(url, filename)
        return filename

def main():#running main function that will test the logic explained above
    # 1. Initialize the classifier
    classifier = CarVsTruck('best.pt') # I am creating an instance of the class (object) and input the path of the model, model weights. 
    
    # 2. Test with a URL (or a local path)
    test_source = "https://images.pexels.com/photos/28264507/pexels-photo-28264507.jpeg"
    
    # FIX: Check if it's a URL OR a local file path
    if test_source.startswith("http") or os.path.exists(test_source):
        predicted_class = classifier.predict(test_source)
        logging.info(f"This is an image of a: {predicted_class}")# will tell us the type of class based on the image
    else:
        logging.error(f"Source {test_source} was not found on your computer or is not a valid URL.")
