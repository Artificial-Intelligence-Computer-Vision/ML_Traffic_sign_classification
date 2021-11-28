from header_imports import *


class computer_vision_transfer_learning(computer_vision_building):
    def __init__(self, currently_build_model, model_type, image_type, category)):

        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.currently_build_model = self.path_to_model + currently_build_model
        self.model.load_weights(self.currently_build_model)


        if self.model_type == "model1":
            self.create_models_1()
        elif self.model_type == "model2":
            self.create_models_2()
        elif self.model_type == "model3":
            self.create_model_3()

        self.save_model_summary()

        
    def create_models_1(self):

        self.model.add(Conv2D(filters=64,kernel_size=(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        self.model.add(MaxPooling2D(pool_size = (4,4)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=32,kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=16,kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(MaxPooling2D(pool_size = (1,1)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))


        for layer in custom_vgg_model.layers[:-1]:
	        layer.trainable = False

        custom_vgg_model.layers[3].trainable


        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return self.model


    
    def create_models_2(self):

        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape = self.input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units = self.number_classes, activation="softmax"))
        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
	
        return self.model


    def create_model_3(self):

        self.MyConv(first = True)
        self.MyConv()
        self.MyConv()
        self.MyConv()
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        self.model.compile(loss = "binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])
        
        return self.model


    def MyConv(self, first = False):

        if first == False:
            self.model.add(Conv2D(64, (4, 4),strides = (1,1), padding="same",
                input_shape = self.input_shape))
        else:
            self.model.add(Conv2D(64, (4, 4),strides = (1,1), padding="same",
                 input_shape = self.input_shape))
    
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(32, (4, 4),strides = (1,1),padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.25))



 def save_model_summary(self):

        with open(self.model_summary + self.create_model_type +"_summary_architecture_transfer_learning_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()
