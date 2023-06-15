import torch
import numpy as np

biased_data = np.loadtxt('../../datasets/kuaishou/biased_data.txt').astype(int)
uni_data = np.loadtxt('../../datasets/kuaishou/uni_data.txt').astype(int)
print('biased_data', biased_data)
print('uni_data', uni_data)

num_bi_unique_users = len(np.unique(biased_data[:, 0]))
num_bi_unique_items = len(np.unique(biased_data[:, 1]))

num_un_unique_users = len(np.unique(uni_data[:, 0]))
num_un_unique_items = len(np.unique(uni_data[:, 1]))

print('num_bi_unique_users', num_bi_unique_users)
print('num_bi_unique_items', num_bi_unique_items)
print('num_un_unique_users', num_un_unique_users)
print('num_un_unique_items', num_un_unique_items)

print('np.max(biased_data[:, 0])', np.max(biased_data[:, 0]))
print('np.max(biased_data[:, 1])', np.max(biased_data[:, 1]))
print('uni_data[:, 0]', np.max(uni_data[:, 0]))
print('uni_data[:, 1]', np.max(uni_data[:, 1]))

b_inter_u = []
u_inter_u = []

count = 0
for user in range(np.max(biased_data[:, 0])+1):
    print('user', user)
    num_inter_per_user = (biased_data[:, 0] == user).sum()
    if num_inter_per_user != 0:
        b_inter_u.append(biased_data[biased_data[:, 0]==user])
        u_inter_u.append(uni_data[uni_data[:, 0]==user])

print('len(b_inter_u)', len(b_inter_u))
print('len(u_inter_u)', len(u_inter_u))
for i in range(len(b_inter_u)):
    b_inter_u[i][:, 0] = i
    u_inter_u[i][:, 0] = i


b_inter_u_array = np.vstack(b_inter_u)
u_inter_u_array = np.vstack(u_inter_u)
print('np.max(b_inter_u_array[:, 0])', np.max(b_inter_u_array[:, 0]))
print('u_inter_u_array[:, 0]', np.max(u_inter_u_array[:, 0]))
print('len(np.unique(b_inter_u_array[:, 0]))', len(np.unique(b_inter_u_array[:, 0])))
print('len(np.unique(u_inter_u_array[:, 0]))', len(np.unique(u_inter_u_array[:, 0])))




b_inter_ui = []
u_inter_ui = []

count = 0
for item in range(np.max(b_inter_u_array[:, 1])+1):
    num_inter_per_item = (b_inter_u_array[:, 1] == item).sum()
    if num_inter_per_item != 0:
        b_inter_ui.append(b_inter_u_array[b_inter_u_array[:, 1]==item])
        u_inter_ui.append(u_inter_u_array[u_inter_u_array[:, 1]==item])

print('len(b_inter_ui)', len(b_inter_ui))
print('len(u_inter_ui)', len(u_inter_ui))
for i in range(len(b_inter_ui)):
    b_inter_ui[i][:, 1] = i
    u_inter_ui[i][:, 1] = i

b_inter_ui_array = np.vstack(b_inter_ui)
u_inter_ui_array = np.vstack(u_inter_ui)
print('np.max(b_inter_ui_array[:, 1])', np.max(b_inter_ui_array[:, 1]))
print('u_inter_ui_array[:, 1]', np.max(u_inter_ui_array[:, 1]))
print('len(np.unique(b_inter_ui_array[:, 1]))', len(np.unique(b_inter_ui_array[:, 1])))
print('len(np.unique(u_inter_ui_array[:, 1]))', len(np.unique(u_inter_ui_array[:, 1])))

np.savetxt('../../datasets/kuaishou/b_inter_ui_array.txt', b_inter_ui_array, fmt='%d')
np.savetxt('../../datasets/kuaishou/u_inter_ui_array.txt', u_inter_ui_array, fmt='%d')
