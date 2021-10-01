from header_imports import *
from computer_vision_model_training import *

if __name__ == "__main__":
    
    # Begin analysis
    if len(sys.argv) != 1:

        # Build the model
        if sys.argv[1] == "model_building":
            computer_vision__analysis_obj = computer_vision_building(model_type = sys.argv[2], image_type = sys.argv[3], category = sys.argv[4])

        # Classify the images
        if sys.argv[1] == "model_training":
            computer_vision_analysis_obj = computer_vision_training(model_type = sys.argv[2], image_type = sys.argv[3], category = sys.argv[4])

        # Localization and Detection 
        if sys.arg[1] == "Localization and Detection":
            pass
