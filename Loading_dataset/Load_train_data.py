# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 02:59:53 2020

@author: Sreeraman
"""

try:

    import tensorflow as tf
    import cv2
    import os
    import pickle
    import numpy as np
    print("Library Loaded Successfully ..........")
except:
    print("Library not Found ! ")


class MasterImage(object):

    def __init__(self,PATH='', IMAGE_SIZE = 224):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_data = []
        self.x_train = []
        self.y_train = []
        self.CATEGORIES = []

        # This will get List of categories
        self.list_categories = []

    def get_categories(self):
        for path in os.listdir(self.PATH):
            if '.DS_Store' in path:
                pass
            else:
                self.list_categories.append(path)
        print("Found Categories ",self.list_categories,'\n')
        return self.list_categories

    def Process_Image(self):
        try:
            """
            Return Numpy array of image
            :return: x_train, y_train
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:        # if any image is corrupted
                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_UNCHANGED) #IMREAD_UNCHANGED                 # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize,class_index])
                    except:
                        pass

            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_train.append(x[0])        # Get the x_train
                self.y_train.append(x[1])        # get the label

            x_train = np.asarray(self.x_train) / (255.0)      # Normalize Data
            y_train = np.asarray(self.y_train)

            # reshape x_train

            x_train = x_train.reshape(-3, self.IMAGE_SIZE, self.IMAGE_SIZE, 3) # change to 3 for color and 1 for grey

            return x_train, y_train
        except:
            print("Failed to run Function Process Image ")

    def pickle_image(self):

        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        x_train,y_train = self.Process_Image()

        # Write the Entire Data into a Pickle File
        pickle_out = open('x_train','wb')
        pickle.dump(x_train, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('y_train', 'wb')
        pickle.dump(y_train, pickle_out)
        pickle_out.close()

        print("Pickled Image Successfully ")
        return x_train,y_train

    def load_dataset(self):

        try:
            # Read the Data from Pickle Object
            X_Temp = open('x_train','rb')
            x_train = pickle.load(X_Temp)

            Y_Temp = open('y_train','rb')
            y_train = pickle.load(Y_Temp)

            print('Reading Dataset from PIckle Object')

            return x_train,y_train

        except:
            print('Could not Found Pickle File ')
            print('Loading File and Dataset  ..........')

            x_train,y_train = self.pickle_image()
            return x_train,y_train


if __name__ == "__main__":
    path = 'D:\Sree program\seafloor_data_new/train'
    a = MasterImage(PATH=path,
                    IMAGE_SIZE=224)

    x_train,y_train = a.load_dataset()
    print(x_train.shape)


















