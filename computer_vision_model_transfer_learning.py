from header_imports import *


class computer_vision_transfer_learning(object):
    def __init__(self, currently_build_model):

        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.currently_build_model = self.path_to_model + currently_build_model
        
        self.prediction_config = transfer_learning_config()
        self.model = MaskRCNN(mode='training', model_dir=self.save_model_path, config=self.prediction_config)
        self.model.load_weights(self.currently_build_model, by_name=True)
        

