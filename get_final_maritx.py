import numpy as np
import os
import math

# process ad_maritx

file=open(r"ad_matrix\ad_martix.txt")
file_list=file.readlines()
length=len(file_list)
to_martix=np.zeros((length,7))

index=0
file=open(r"ad_matrix\ad_martix.txt")

thshold=0.4
p=0.25

for data in file.readlines():
    data = data.strip('\n')
    nums = data.split(" ")
    x_nums=[]
    for x in nums:
        if float(x) <thshold:
            x=0
        else:
            x=1
        x_nums.append(float(x))
    to_martix[index,:] = x_nums[:]
    index +=1

print(to_martix)
print(type(to_martix))

adj=to_martix*p/(to_martix.sum(0,keepdims=True)+1e-6)
adj=adj+np.identity(7,np.int)
print("adj",adj)

D=np.power(adj.sum(1),-0.5)
D=np.diag(D)
adj_martix=np.matmul(np.matmul(D,adj),D)

print(to_martix)
print(adj_martix)
np.save("final_maritx.npy",adj_martix)
