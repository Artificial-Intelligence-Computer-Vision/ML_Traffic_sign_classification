from header_imports import *

class computer_vision_localization_detection(object):
    def __init__(self, currently_build_model = "normal_category_1_model1_computer_vision_categories_43_model.h5"):
        
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

	    self.category_names = traffic_sign_categories.category_names
        self.category_names_1 = traffic_sign_categories.category_names_1            			
        self.category_names_2 = traffic_sign_categories.category_names_2            			
        self.category_names_3 = traffic_sign_categories.category_names_3

        self.category == "category_1"

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

        file_names = next(os.walk(self.image_path))[2]
        
        print(file
        image = skimage.io.imread(os.path.join(self.image_path, random.choice(file_names)))
        r = model.detect([image], verbose=1)[0]

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            self.category_names, r['scores'])


