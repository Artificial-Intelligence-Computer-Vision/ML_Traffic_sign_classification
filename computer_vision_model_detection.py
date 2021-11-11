from header_import import *

class computer_vision_localization_detection(computer_vision_transfer_learning):
    def __init__(self,currently_build_model):
        super().__init__(currently_build_model)
        
        self.boxes = None
        self.class_ids = list()
        self.image_path = "traffic_signs" + "/Test"




