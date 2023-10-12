import os
import numpy as np

root_path = "./seed_output"
dataset = "coat2"
model = "directau"

full_path = os.path.join(root_path, dataset)
full_path = os.path.join(full_path, model)

gcn_path = full_path + "_seed"
mf_path = full_path + "_seed_gcn"

# print(full_path)
full_matrix = np.empty((0,6), int)

with open(mf_path, "r") as f:
    line = f.readline()
    line_num=1
    while line:
        if line_num%9 == 5 and line_num!=5:
            num_ls = line.strip().split(",")
            full_matrix = np.append(full_matrix, np.array([num_ls]), axis=0)
            # print(line.strip().split(","))
        line = f.readline()
        line_num+=1
# full_matrix = np.array(full_matrix)
# print(full_matrix.shape)
full_matrix = full_matrix.astype(np.float32)
final_ls = full_matrix.sum(axis=0) / 5
print(str(final_ls[0])+"\t"+str(final_ls[1])+"\t"+str(final_ls[2])+"\t"+str(final_ls[3])+"\t"+str(final_ls[4])+"\t"+str(final_ls[5]))


full_matrix = np.empty((0,6), int)

with open(gcn_path, "r") as f:
    line = f.readline()
    line_num=1
    while line:
        if line_num%9 == 5 and line_num!=5:
            num_ls = line.strip().split(",")
            full_matrix = np.append(full_matrix, np.array([num_ls]), axis=0)
            # print(line.strip().split(","))
        line = f.readline()
        line_num+=1
# full_matrix = np.array(full_matrix)
# print(full_matrix.shape)
full_matrix = full_matrix.astype(np.float32)
final_ls = full_matrix.sum(axis=0) / 5
print(str(final_ls[0])+"\t"+str(final_ls[1])+"\t"+str(final_ls[2])+"\t"+str(final_ls[3])+"\t"+str(final_ls[4])+"\t"+str(final_ls[5]))



# print(sum_ls)