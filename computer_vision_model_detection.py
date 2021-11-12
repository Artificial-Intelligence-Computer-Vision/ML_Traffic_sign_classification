from header_imports import *

class computer_vision_localization_detection(object):
    def __init__(self,currently_build_model = "normal_category_1_model1_computer_vision_categories_43_model.h5"):
        
        self.boxes = None
        self.class_ids = list()
        self.number_images_to_plot = 20
        self.image_path = "traffic_signs" + "/Test"

        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.currently_build_model = self.path_to_model + currently_build_model

        self.images = [count for count in glob(self.image_path +'*') if 'png' in count]
        
        self.prediction_config = Transfer_Learning_Config()
        self.model = MaskRCNN(mode='inference', model_dir=self.save_model_path, config=self.prediction_config)
        self.model.load_weights(self.currently_build_model, by_name=True)

        self.category_names = ["Speed limit (20km/h)",
            			"Speed limit (30km/h)", 
            			"Speed limit (50km/h)", 
            			"Speed limit (60km/h)", 
            			"Speed limit (70km/h)", 
            			"Speed limit (80km/h)", 
            			"End of speed limit (80km/h)", 
            			"Speed limit (100km/h)", 
            			"Speed limit (120km/h)", 
            			"No passing", 
            			"No passing for vehicles over 3.5 metric tons", 
            			"Right-of-way at intersection", 
            			"Priority road", 
            			"Yield", 
            			"Stop", 
            			"No vehicles", 
            			"Vehicles over 3.5 metric tons prohibited", 
            			"No entry", 
            			"General caution", 
            			"Dangerous curve left", 
            			"Dangerous curve right", 
            			"Double curve", 
            			"Bumpy road", 
            			"Slippery road", 
            			"Road narrows on the right", 
            			"Road work", 
            			"Traffic signals", 
            			"Pedestrians", 
            			"Children crossing", 
            			"Bicycles crossing", 
            			"Beware of ice/snow",
            			"Wild animals crossing", 
            			"End speed + passing limits", 
            			"Turn right ahead", 
            			"Turn left ahead", 
            			"Ahead only", 
            			"Go straight or right", 
            			"Go straight or left", 
            			"Keep right", 
            			"Keep left", 
            			"Roundabout mandatory", 
            			"End of no passing", 
            			"End of no passing by vehicles over 3.5 metric tons"]
            			
        self.category_names_1 = ["Speed limit (20km/h)",
            			"Speed limit (30km/h)", 
            			"Speed limit (50km/h)", 
            			"Speed limit (60km/h)", 
            			"Speed limit (70km/h)", 
            			"Speed limit (80km/h)", 
            			"End of speed limit (80km/h)", 
            			"Speed limit (100km/h)", 
            			"Speed limit (120km/h)", 
            			"No passing", 
            			"No passing for vehicles over 3.5 metric tons", 
            			"Right-of-way at intersection", 
            			"Priority road", 
            			"Yield", 
            			"Stop"]
            			
            			
        self.category_names_2 = ["No vehicles", 
            			"Vehicles over 3.5 metric tons prohibited", 
            			"No entry", 
                        "General caution", 
            			"Dangerous curve left", 
            			"Dangerous curve right", 
            			"Double curve", 
            			"Bumpy road", 
            			"Slippery road", 
            			"Road narrows on the right", 
            			"Road work", 
            			"Traffic signals", 
            			"Pedestrians", 
            			"Children crossing"]
            			
            			
        self.category_names_3 = ["Bicycles crossing", 
            			"Beware of ice/snow",
            			"Wild animals crossing", 
            			"End speed + passing limits", 
            			"Turn right ahead", 
            			"Turn left ahead", 
            			"Ahead only", 
            			"Go straight or right", 
            			"Go straight or left", 
            			"Keep right", 
            			"Keep left", 
            			"Roundabout mandatory", 
            			"End of no passing", 
            			"End of no passing by vehicles over 3.5 metric tons"]
      

        if self.category == "category_1":
            self.category_names = self.category_names_1
        elif self.category == "category_2":
            self.category_names = self.category_names_2
        elif self.category == "category_3":
            self.category_names = self.category_names_3
        elif self.category == "normal":
            self.category_names = self.categories
        elif self.category == "regular":
            self.category_names = self.category_names


        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
        results = model.detect([image], verbose=1)

        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            self.category_names, r['scores'])


