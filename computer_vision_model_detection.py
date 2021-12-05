from header_imports import *

detection_name =  "Detection Traffic Signs"

class detection_config(Config):
    NAME = detection_name
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 43 + 1
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.7

config = detection_config()

class detection(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class computer_vision_localization_detection(utils.Dataset):
    def __init__(self, category, currently_build_model = "normal_category_1_model1_computer_vision_categories_43_model.h5"):
        
        self.boxes = None
        self.class_ids = list()
        self.number_images_to_plot = 20
        self.image_path = "traffic_signs" + "/Test"
        self.category = category
        
        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.save_model_path = os.path.join(self.save_model_path, "logs")
        self.currently_build_model = self.path_to_model + currently_build_model

        self.prediction_config = detection()
        self.model = modellib.MaskRCNN(mode='inference', model_dir=self.save_model_path, config=self.prediction_config)
        self.model.load_weights(self.currently_build_model, by_name=True) 

        self.category_names = traffic_sign_categories.category_names
        self.category_names_1 = traffic_sign_categories.category_names_1            			
        self.category_names_2 = traffic_sign_categories.category_names_2            			
        self.category_names_3 = traffic_sign_categories.category_names_3
        self.categories = traffic_sign_categories.categories

        for i in range(0, 43):
            self.add_class(detection_name, i, self.category_names[i])

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

        self.localization_detection()



    def localization_detection(self):

        count = os.listdir(self.image_path)
        for i in range(0,len(count)):
            path = os.path.join(self.image_path, count[i])
            
            if os.path.isfile(path):
                file_names = next(os.walk(self.image_path))[2]
                image = skimage.io.imread(os.path.join(self.image_path, count[i]))
            
                detect = model.detect([image], verbose=1)[0]
                visualize.display_instances(count[i],image, detect['rois'], detect['masks'], detect['class_ids'], self.category_names, detect['scores'])

    

    def load_custom(self, dataset_dir, subset):

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open('D:/MaskRCNN-aar/Dataset/train/demo_json.json'))

        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
           
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            name_dict = {"laptop": 1,"tab": 2,"phone": 3}
            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("object", image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons, num_ids=num_ids)

    def load_mask(self, image_id):
       
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
