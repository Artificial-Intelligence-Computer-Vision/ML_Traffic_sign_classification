from header_imports import *

class Transfer_Learning_Config(Config):
    NAME = "Transfer Learning Traffic Signs"
    NUM_CLASSES = 43 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256 


class computer_vision_transfer_learning(object):
    def __init__(self, currently_build_model):

        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.currently_build_model = self.path_to_model + currently_build_model
        
        self.prediction_config = Transfer_Learning_Config()
        self.model = MaskRCNN(mode='training', model_dir=self.save_model_path, config=self.prediction_config)
        self.model.load_weights(self.currently_build_model, by_name=True)
        

