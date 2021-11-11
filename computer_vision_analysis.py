from header_imports import *

if __name__ == "__main__":
    
    # Begin analysis
    if len(sys.argv) != 1:

        # Build the model
        if sys.argv[1] == "model_building":
            computer_vision__analysis_obj = computer_vision_building(model_type = sys.argv[2], image_type = sys.argv[3], category = sys.argv[4])

        # Classify the images
        if sys.argv[1] == "model_training":
            computer_vision_analysis_obj = computer_vision_training(model_type = sys.argv[2], image_type = sys.argv[3], category = sys.argv[4])
        
        # Indentify and image with model
        if sys.argv[1] == "image_classification_with_model":
            pass

        # Localization and Detection 
        if sys.argv[1] == "localization_and_detection":
            pass

        # Transfer Learning 
        if sys.argv[1] == "transfer_learning":
            pass


