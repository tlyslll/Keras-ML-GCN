import numpy as np

#

file1=open(r"word_embding\word_embding.txt")
file2=open(r"train\train.txt")

num_class=7
num_label=3
embding_dimon=300
total_label_list=[]
label_list=[]
next_label_list=[]
class_list=[i for i in range(num_class)]
temp=0
emdbing_data_list=[]


for embding_data in file1.readlines():
    embding_data=embding_data.strip("\n")
    embding_data = embding_data.split(",")
    for j in embding_data[1].split(" "):
        emdbing_data_list.append(float(j))


print("emdbing_data_list=",emdbing_data_list)
print(len(emdbing_data_list))

emdbing_data_matrix=np.array(emdbing_data_list)
emdbing_data_matrix=emdbing_data_matrix.reshape((num_class,embding_dimon))

print("emdbing_data_matrix=",emdbing_data_matrix)
print(emdbing_data_matrix.shape)

for data in file2.readlines():
    data=data.strip("\n")
    data=data.split(" ")
    temp_data=data[1:num_label+1]
    for j in temp_data:
        if j!="NULL":
            label_list.append(int(j))

    if next_label_list:
        pass
    else:
        for j in label_list:
            next_label_list.append(j)
        print("first next_label_list=",next_label_list)
        emdbing=np.zeros((len(class_list),embding_dimon))
        for x in next_label_list:
            emdbing[x]=emdbing_data_matrix[x]
        print("emdbing=", emdbing)
        np.save(r"word_embding/" + str(next_label_list) + ".npy", emdbing)

    if next_label_list == label_list:
        pass
    else:
        print('label_list=', label_list)
        next_label_list=[]
        for j in label_list:
            next_label_list.append(j)
        print("next_label_list after change=",next_label_list)

        emdbing=np.zeros((len(class_list),embding_dimon))
        for x in next_label_list:
            emdbing[x]=emdbing_data_matrix[x]
        print("emdbing=",emdbing)
        np.save(r"word_embding/"+str(next_label_list)+".npy",emdbing)
    label_list=[]

