import os
import numpy as np

#combine_word_vector

path="word_embding"
data_list=os.listdir(path)
num_class=7
feature_dim=300
matrix=np.zeros((num_class,feature_dim))
num_file=0

for i in data_list:

    if i[-4:]==".npy":
        num_file += 1
        data=np.load(path+"/"+str(i))

        matrix=matrix+data


matrix=matrix/num_file

np.save(path+"/combine_word_embding.npy",matrix)