from header_imports import *

class traffic_sign_categories:

    category_names = ["Speed limit (20km/h)",
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

    category_names_1 = ["Speed limit (20km/h)",
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

    category_names_2 = ["No vehicles", 
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
            			
    category_names_3 = ["Bicycles crossing", 
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





class transfer_learning_config(Config):
    name = "transfer learning traffic signs"
    num_classes = 43 + 1
    gpu_count = 1
    images_per_gpu = 1
    image_min_dim = 256
    image_max_dim = 256 

