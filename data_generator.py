from tensorflow.keras.utils import Sequence
import math
import numpy as np
import matplotlib.image as mpimg
import scipy.sparse as sp
import cv2
import tensorflow as tf

class DataGenerator(Sequence):

    def __init__(self, list_IDs, batch_size=1, img_size=(224, 224), *args, **kwargs):
        """
           self.list_IDs:a list of all image file names that you want to train
           self.batch_size:For each batch generation, train the sample size
           self.img_size:Picture size of training。
        """

        self.list_IDs = list_IDs

        self.word_embding_dir=list_IDs
        self.batch_size = batch_size
        self.img_size = img_size

        self.on_epoch_end()


    def __len__(self):
        """
           Returns the length of the generator, that is, the total number of batches of data generated

        """
        return int(math.ceil(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        """
           returns the processed data each time we need it
        """

        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k,] for k in indices]

        image,word_embding,adj_martix,label = self.__data_generation(list_IDs_temp)
        return {"image":image,"word_embding":word_embding,"adj_martix":adj_martix},label

    def on_epoch_end(self):
        """
           This function will be executed automatically at the end of each epoch while training, in this case randomly shuffling the index order to facilitate the next batch run

        """
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)


    def __data_generation(self, list_IDs_temp):
        """
           Given the file name, generate data
        """
        X = np.empty((self.batch_size, *self.img_size, 3))
        Y = np.empty((self.batch_size, 7), dtype=np.float32)
        word_embding=np.empty((self.batch_size,7,300),dtype=np.float32) # 7 is number of label in our dataset, 300 is dimension of word vector
        adj_matrix = np.load("ad_matrix/final_maritx.npy") # need to modify in your task

        adj_matrix=adj_matrix.reshape((49,)) # 49 is 7*7, (49,)
        new_adj_matrix=np.empty((self.batch_size,49,))

        for i,x in enumerate(list_IDs_temp): #one_hot label
            X[i,]=cv2.imread(x[0])
            label=np.zeros((7,))
            for j in x[1:-1]:
                if j!="NULL": # NULL is no label
                    label[int(j)]=1

            Y[i,]=label
            graph_feature = np.load(x[-1])
            word_embding[i,]=graph_feature
            new_adj_matrix[i,]=adj_matrix

        X=X/255

        return X,word_embding,new_adj_matrix,Y