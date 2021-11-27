from header_imports import *

class prediction_with_model(object):
    def __init__(self, model, image_path):

        self.image_path = image_path
        self.model = model

        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (224, 224))
        dims = np.expand_dims(img, axis=0)
        dims = preprocess_input(dims)

        preds=model.predict(dims)
        preds
