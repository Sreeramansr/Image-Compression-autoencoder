# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 02:59:49 2020

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
        self.x_test = []
        self.y_test = []
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
            :return: x_test, y_test
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:        # if any image is corrupted
                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_UNCHANGED)                 # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize,class_index])
                    except:
                        pass

            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_test.append(x[0])        # Get the x_test
                self.y_test.append(x[1])        # get the label

            x_test = np.asarray(self.x_test) / (255.0)      # Normalize Data
            y_test = np.asarray(self.y_test)

            # reshape x_test

            x_test = x_test.reshape(-3, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)

            return x_test, y_test
        except:
            print("Failed to run Function Process Image ")

    def pickle_image(self):

        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        x_test,y_test = self.Process_Image()

        # Write the Entire Data into a Pickle File
        pickle_out = open('x_test','wb')
        pickle.dump(x_test, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('y_test', 'wb')
        pickle.dump(y_test, pickle_out)
        pickle_out.close()

        print("Pickled Image Successfully ")
        return x_test,y_test

    def load_dataset(self):

        try:
            # Read the Data from Pickle Object
            X_Temp = open('x_test','rb')
            x_test = pickle.load(X_Temp)

            Y_Temp = open('y_test','rb')
            y_test = pickle.load(Y_Temp)

            print('Reading Dataset from PIckle Object')

            return x_test,y_test

        except:
            print('Could not Found Pickle File ')
            print('Loading File and Dataset  ..........')

            x_test,y_test = self.pickle_image()
            return x_test,y_test


if __name__ == "__main__":
    path = 'D:\Sree program\seafloor_data_new/test'
    b = MasterImage(PATH=path,
                    IMAGE_SIZE=224)

    x_test,y_test = b.load_dataset()
    print(x_test.shape)


















