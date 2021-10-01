from header_imports import *

class computer_vision_building(object):

    def __init__(self, model_type, image_type, category):


        self.images = []
        self.filename = []
        self.image_file = []
        # 0 for False and 1 for True for label name
        self.label_name = []
        self.number_classes = 43
        self.image_size = 240
        self.path  = "traffic_signs/"
        self.image_type = image_type
        self.category = category

        # Determine
        if self.image_type == "small_traffic_sign":
            self.true_path = self.path + "Small_Traffic_Sign/"
        elif self.image_type == "regular":
            self.true_path = self.path + "Train/"
        elif self.image_type == "train1":
            self.true_path = self.path + "Train_1_50/"
        elif self.image_type == "train2":
            self.true_path = self.path + "Train_2_25/"
        elif self.image_type == "train3":
            self.true_path = self.path + "Train_3_25/"

        
        self.valid_images = [".jpg",".png"]
        self.input_shape = None
        self.advanced_categories = ["0", "1", "2", "2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30","31", "32", "33", "34", "35", "36", "37", "38","39", "40", "41", "42"]
        
        self.categories = ["One Way Right", "Slow Xing", "Yield", "One Way Left", "Traffic Light Sign", "Stop", "Ducky"]

        self.advanced_categories_1 = ["0", "1", "2", "2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13", "14"]
        self.advanced_categories_2 = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]
        self.advanced_categories_3 = ["29", "30","31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42"]
        
        self.category_names = ["Speed limit (20km/h)",
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
            			
        self.category_names_1 = ["Speed limit (20km/h)",
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
            			
            			
        self.category_names_2 = ["No vehicles", 
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
            			
            			
        self.category_names_3 = ["Bicycles crossing", 
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
	
	
        # Split training data variables
        self.X_train = None
        self.X_test = None
        self.Y_train_vec = None
        self.Y_test_vec = None

        # model informations
        self.model = None
        
        # model summary path 
        self.model_summary = "model_summary/"

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.create_model_type = model_type
        
        if self.category == "category_1":
	    # Check validity
            self.check_valid(self.advanced_categories_1[0])
            self.check_valid(self.advanced_categories_1[1])
            self.check_valid(self.advanced_categories_1[2])
            self.check_valid(self.advanced_categories_1[3])
            self.check_valid(self.advanced_categories_1[4])
            self.check_valid(self.advanced_categories_1[5])
            self.check_valid(self.advanced_categories_1[6])
            self.check_valid(self.advanced_categories_1[7])
            self.check_valid(self.advanced_categories_1[8])
            self.check_valid(self.advanced_categories_1[9])
            self.check_valid(self.advanced_categories_1[10])
            self.check_valid(self.advanced_categories_1[11])
            self.check_valid(self.advanced_categories_1[12])
            self.check_valid(self.advanced_categories_1[13])
            self.check_valid(self.advanced_categories_1[14])
        	
        elif self.category == "category_2":
	    # Check validity
            self.check_valid(self.advanced_categories_2[0])
            self.check_valid(self.advanced_categories_2[1])
            self.check_valid(self.advanced_categories_2[2])
            self.check_valid(self.advanced_categories_2[3])
            self.check_valid(self.advanced_categories_2[4])
            self.check_valid(self.advanced_categories_2[5])
            self.check_valid(self.advanced_categories_2[6])
            self.check_valid(self.advanced_categories_2[7])
            self.check_valid(self.advanced_categories_2[8])
            self.check_valid(self.advanced_categories_2[9])
            self.check_valid(self.advanced_categories_2[10])
            self.check_valid(self.advanced_categories_2[11])
            self.check_valid(self.advanced_categories_2[12])
            self.check_valid(self.advanced_categories_2[13])

        elif self.category == "category_3":
	    # Check validity
            self.check_valid(self.advanced_categories_3[0])
            self.check_valid(self.advanced_categories_3[1])
            self.check_valid(self.advanced_categories_3[2])
            self.check_valid(self.advanced_categories_3[3])
            self.check_valid(self.advanced_categories_3[4])
            self.check_valid(self.advanced_categories_3[5])
            self.check_valid(self.advanced_categories_3[6])
            self.check_valid(self.advanced_categories_3[7])
            self.check_valid(self.advanced_categories_3[8])
            self.check_valid(self.advanced_categories_3[9])
            self.check_valid(self.advanced_categories_3[10])
            self.check_valid(self.advanced_categories_3[11])
            self.check_valid(self.advanced_categories_3[12])
            self.check_valid(self.advanced_categories_3[13])

        elif self.category == "regular":
            # Check validity
            self.check_valid(self.advanced_categories[0])
            self.check_valid(self.advanced_categories[1])
            self.check_valid(self.advanced_categories[2])
            self.check_valid(self.advanced_categories[3])
            self.check_valid(self.advanced_categories[4])
            self.check_valid(self.advanced_categories[5])
            self.check_valid(self.advanced_categories[6])
            self.check_valid(self.advanced_categories[7])
            self.check_valid(self.advanced_categories[8])
            self.check_valid(self.advanced_categories[9])
            self.check_valid(self.advanced_categories[10])
            self.check_valid(self.advanced_categories[11])
            self.check_valid(self.advanced_categories[12])
            self.check_valid(self.advanced_categories[13])
            self.check_valid(self.advanced_categories[14])
            self.check_valid(self.advanced_categories[15])
            self.check_valid(self.advanced_categories[16])
            self.check_valid(self.advanced_categories[17])
            self.check_valid(self.advanced_categories[18])
            self.check_valid(self.advanced_categories[19])
            self.check_valid(self.advanced_categories[20])
            self.check_valid(self.advanced_categories[21])
            self.check_valid(self.advanced_categories[22])
            self.check_valid(self.advanced_categories[23])
            self.check_valid(self.advanced_categories[24])
            self.check_valid(self.advanced_categories[25])
            self.check_valid(self.advanced_categories[26])
            self.check_valid(self.advanced_categories[27])
            self.check_valid(self.advanced_categories[28])
            self.check_valid(self.advanced_categories[29])
            self.check_valid(self.advanced_categories[30])
            self.check_valid(self.advanced_categories[31])
            self.check_valid(self.advanced_categories[32])
            self.check_valid(self.advanced_categories[33])
            self.check_valid(self.advanced_categories[34])
            self.check_valid(self.advanced_categories[35])
            self.check_valid(self.advanced_categories[36])
            self.check_valid(self.advanced_categories[37])
            self.check_valid(self.advanced_categories[38])
            self.check_valid(self.advanced_categories[39])
            self.check_valid(self.advanced_categories[40])
            self.check_valid(self.advanced_categories[41])
            self.check_valid(self.advanced_categories[42])

        elif self.category == "normal":
            # Check validity
            self.check_valid(self.categories[0])
            self.check_valid(self.categories[1])
            self.check_valid(self.categories[2])
            self.check_valid(self.categories[3])
            self.check_valid(self.categories[4])
            self.check_valid(self.categories[5])
            self.check_valid(self.categories[6])

		



        if self.category == "category_1":
	    # Resize image
            self.resize_image_and_label_image(self.advanced_categories_1[0])
            self.resize_image_and_label_image(self.advanced_categories_1[1])
            self.resize_image_and_label_image(self.advanced_categories_1[2])
            self.resize_image_and_label_image(self.advanced_categories_1[3])
            self.resize_image_and_label_image(self.advanced_categories_1[4])
            self.resize_image_and_label_image(self.advanced_categories_1[5])
            self.resize_image_and_label_image(self.advanced_categories_1[6])
            self.resize_image_and_label_image(self.advanced_categories_1[7])
            self.resize_image_and_label_image(self.advanced_categories_1[8])
            self.resize_image_and_label_image(self.advanced_categories_1[9])
            self.resize_image_and_label_image(self.advanced_categories_1[10])
            self.resize_image_and_label_image(self.advanced_categories_1[11])
            self.resize_image_and_label_image(self.advanced_categories_1[12])
            self.resize_image_and_label_image(self.advanced_categories_1[13])
            self.resize_image_and_label_image(self.advanced_categories_1[14])
        	
        elif self.category == "category_2":
	    # Resize image
            self.resize_image_and_label_image(self.advanced_categories_2[0])
            self.resize_image_and_label_image(self.advanced_categories_2[1])
            self.resize_image_and_label_image(self.advanced_categories_2[2])
            self.resize_image_and_label_image(self.advanced_categories_2[3])
            self.resize_image_and_label_image(self.advanced_categories_2[4])
            self.resize_image_and_label_image(self.advanced_categories_2[5])
            self.resize_image_and_label_image(self.advanced_categories_2[6])
            self.resize_image_and_label_image(self.advanced_categories_2[7])
            self.resize_image_and_label_image(self.advanced_categories_2[8])
            self.resize_image_and_label_image(self.advanced_categories_2[9])
            self.resize_image_and_label_image(self.advanced_categories_2[10])
            self.resize_image_and_label_image(self.advanced_categories_2[11])
            self.resize_image_and_label_image(self.advanced_categories_2[12])
            self.resize_image_and_label_image(self.advanced_categories_2[13])

        elif self.category == "category_3":
	    # Resize image
            self.resize_image_and_label_image(self.advanced_categories_3[0])
            self.resize_image_and_label_image(self.advanced_categories_3[1])
            self.resize_image_and_label_image(self.advanced_categories_3[2])
            self.resize_image_and_label_image(self.advanced_categories_3[3])
            self.resize_image_and_label_image(self.advanced_categories_3[4])
            self.resize_image_and_label_image(self.advanced_categories_3[5])
            self.resize_image_and_label_image(self.advanced_categories_3[6])
            self.resize_image_and_label_image(self.advanced_categories_3[7])
            self.resize_image_and_label_image(self.advanced_categories_3[8])
            self.resize_image_and_label_image(self.advanced_categories_3[9])
            self.resize_image_and_label_image(self.advanced_categories_3[10])
            self.resize_image_and_label_image(self.advanced_categories_3[11])
            self.resize_image_and_label_image(self.advanced_categories_3[12])
            self.resize_image_and_label_image(self.advanced_categories_3[13])
        
        elif self.category == "regular":
            # Resize image
            self.resize_image_and_label_image(self.advanced_categories[0])
            self.resize_image_and_label_image(self.advanced_categories[1])
            self.resize_image_and_label_image(self.advanced_categories[2])
            self.resize_image_and_label_image(self.advanced_categories[3])
            self.resize_image_and_label_image(self.advanced_categories[4])
            self.resize_image_and_label_image(self.advanced_categories[5])
            self.resize_image_and_label_image(self.advanced_categories[6])
            self.resize_image_and_label_image(self.advanced_categories[7])
            self.resize_image_and_label_image(self.advanced_categories[8])
            self.resize_image_and_label_image(self.advanced_categories[9])
            self.resize_image_and_label_image(self.advanced_categories[10])
            self.resize_image_and_label_image(self.advanced_categories[11])
            self.resize_image_and_label_image(self.advanced_categories[12])
            self.resize_image_and_label_image(self.advanced_categories[13])
            self.resize_image_and_label_image(self.advanced_categories[14])
            self.resize_image_and_label_image(self.advanced_categories[15])
            self.resize_image_and_label_image(self.advanced_categories[16])
            self.resize_image_and_label_image(self.advanced_categories[17])
            self.resize_image_and_label_image(self.advanced_categories[18])
            self.resize_image_and_label_image(self.advanced_categories[19])
            self.resize_image_and_label_image(self.advanced_categories[20])
            self.resize_image_and_label_image(self.advanced_categories[21])
            self.resize_image_and_label_image(self.advanced_categories[22])
            self.resize_image_and_label_image(self.advanced_categories[23])
            self.resize_image_and_label_image(self.advanced_categories[24])
            self.resize_image_and_label_image(self.advanced_categories[25])
            self.resize_image_and_label_image(self.advanced_categories[26])
            self.resize_image_and_label_image(self.advanced_categories[27])
            self.resize_image_and_label_image(self.advanced_categories[28])
            self.resize_image_and_label_image(self.advanced_categories[29])
            self.resize_image_and_label_image(self.advanced_categories[30])
            self.resize_image_and_label_image(self.advanced_categories[31])
            self.resize_image_and_label_image(self.advanced_categories[32])
            self.resize_image_and_label_image(self.advanced_categories[33])
            self.resize_image_and_label_image(self.advanced_categories[34])
            self.resize_image_and_label_image(self.advanced_categories[35])
            self.resize_image_and_label_image(self.advanced_categories[36])
            self.resize_image_and_label_image(self.advanced_categories[37])
            self.resize_image_and_label_image(self.advanced_categories[38])
            self.resize_image_and_label_image(self.advanced_categories[39])
            self.resize_image_and_label_image(self.advanced_categories[40])
            self.resize_image_and_label_image(self.advanced_categories[41])
            self.resize_image_and_label_image(self.advanced_categories[42])

        elif self.category == "normal":
            # Resize image
            self.resize_image_and_label_image(self.categories[0])
            self.resize_image_and_label_image(self.categories[1])
            self.resize_image_and_label_image(self.categories[2])
            self.resize_image_and_label_image(self.categories[3])
            self.resize_image_and_label_image(self.categories[4])
            self.resize_image_and_label_image(self.categories[5])
            self.resize_image_and_label_image(self.categories[6])




        # Numpy array
        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))

        self.splitting_data_normalize()

        if self.create_model_type == "model1":
            self.create_models_1()
        elif self.create_model_type == "model2":
            self.create_models_2()
        elif self.create_model_type == "model3":
            self.create_model_3()

        # Saving model summary
        self.save_model_summary()
        
        print("finished")



    # Checks to see if the image is valid or not
    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    # Resize images
    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)
            if self.category != "normal":

                if input_file == "0":
                    self.label_name.append(0)
                elif input_file == "1":
                    self.label_name.append(1)
                elif input_file == "2":
                    self.label_name.append(2)
                elif input_file == "3":
                    self.label_name.append(3)
                elif input_file == "4":
                    self.label_name.append(4)
                elif input_file == "5":
                    self.label_name.append(5)
                elif input_file == "6":
                    self.label_name.append(6)
                elif input_file == "7":
                    self.label_name.append(7)
                elif input_file == "8":
                    self.label_name.append(8)
                elif input_file == "9":
                    self.label_name.append(9)
                elif input_file == "10":
                    self.label_name.append(10)
                elif input_file == "11":
                    self.label_name.append(11)
                elif input_file == "12":
                    self.label_name.append(12)
                elif input_file == "13":
                    self.label_name.append(13)
                elif input_file == "14":
                    self.label_name.append(14)
                elif input_file == "15":
                    self.label_name.append(15)
                elif input_file == "16":
                    self.label_name.append(16)
                elif input_file == "17":
                    self.label_name.append(17)
                elif input_file == "18":
                    self.label_name.append(18)
                elif input_file == "19":
                    self.label_name.append(19)
                elif input_file == "20":
                    self.label_name.append(20)
                elif input_file == "21":
                    self.label_name.append(21)
                elif input_file == "22":
                    self.label_name.append(22)
                elif input_file == "23":
                    self.label_name.append(23)
                elif input_file == "24":
                    self.label_name.append(24)
                elif input_file == "25":
                    self.label_name.append(25)
                elif input_file == "26":
                    self.label_name.append(26)
                elif input_file == "27":
                    self.label_name.append(27)
                elif input_file == "28":
                    self.label_name.append(28)
                elif input_file == "29":
                    self.label_name.append(29)
                elif input_file == "30":
                    self.label_name.append(30)
                elif input_file == "31":
                    self.label_name.append(31)
                elif input_file == "32":
                    self.label_name.append(32)
                elif input_file == "33":
                    self.label_name.append(33)
                elif input_file == "34":
                    self.label_name.append(34)
                elif input_file == "35":
                    self.label_name.append(35)
                elif input_file == "36":
                    self.label_name.append(36)
                elif input_file == "37":
                    self.label_name.append(37)
                elif input_file == "38":
                    self.label_name.append(38)
                elif input_file == "39":
                    self.label_name.append(39)
                elif input_file == "40":
                    self.label_name.append(40)
                elif input_file == "41":
                    self.label_name.append(41)
                elif input_file == "42":
                    self.label_name.append(42)
                else:
                    print("error")
            else:

                if input_file == "One Way Right":
                    self.label_name.append(0)
                elif input_file == "Slow Xing":
                    self.label_name.append(1)
                elif input_file == "Yield":
                    self.label_name.append(2)
                elif input_file == "One Way Left":
                    self.label_name.append(3)
                elif input_file == "Traffic Light Sign":
                    self.label_name.append(4)
                elif input_file == "Stop":
                    self.label_name.append(5)
                elif input_file == "Ducky":
                    self.label_name.append(6)
                else:
                    print("error")




    # Split training data and testing Data and makes it random and normalized it
    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 0.10, random_state = 42)

        self.input_shape = self.X_train.shape[1:]
        
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)

        # Normalize
        self.X_train = self.X_train.astype("float32")
        self.X_train /= 255
        self.X_test = self.X_test.astype("float32")
        self.X_test /= 255


    def get_model(self):
        return self.model

    def get_data(self):
        return self.X_train , self.Y_train, self.X_test, self.Y_test, self.Y_test_vec

    def get_categories(self):
        # Number of categories
        return self.category_names


    def create_models_1(self):

        self.model = Sequential()

        # First Hitten Layer with 64, 7, 7
        self.model.add(Conv2D(64,(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (4,4)))
        self.model.add(Dropout(0.25))
    
        # Second Hitten Layer 32, 7, 7
        self.model.add(Conv2D(32,(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(0.25))
    
        # Third Hitten Layer 32, 7, 7
        self.model.add(Conv2D(16,(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (1,1)))
        self.model.add(Dropout(0.25))
    
        # last layer, output Layer
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))

        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return self.model


    
    def create_models_2(self):

        self.model = Sequential()

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

        self.model = Sequential()
        
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



    # Save the model summery as a txt file
    def save_model_summary(self):

        with open(self.model_summary + self.create_model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
