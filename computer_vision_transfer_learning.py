from header_import import *


class computer_vision_transfer_learning(object):
    def __init__(self, currently_build_model):

        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.currently_build_model = self.path_to_model + currently_build_model
        
        self.prediction_config = PredictionConfig()
        self.model = MaskRCNN(mode='inference', model_dir='./', config=self.config)
        self.model.load_weights(self.currently_build_model, by_name=True)
        


