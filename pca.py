import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator,FixedLocator
from scipy.interpolate import interp1d
import numpy as np
import os
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def read_h5_(name):
    import h5py
    hidden_list1=[]
    hidden_list2=[]
    with h5py.File(name, 'r') as f:
        # 访问 'data' 组
        data_group = f['data']
        # 遍历所有条目
        for entry_name in data_group:
            entry = data_group[entry_name]
            hidden_state_img1 =torch.tensor(entry['result_image_hidden_states_1'][()],dtype=torch.bfloat16)
            hidden_state_img2 = torch.tensor(entry['result_image_hidden_states_2'][()],dtype=torch.bfloat16)
            hidden_state_text1 = torch.tensor(entry['result_text_hidden_states_1'][()],dtype=torch.bfloat16)
            hidden_state_text2 = torch.tensor(entry['result_text_hidden_states_2'][()],dtype=torch.bfloat16)
            hidden_list1.append(hidden_state_img1)
            hidden_list2.append(hidden_state_text1)
    return hidden_list1,hidden_list2

def read_h5_single_(name):
    import h5py
    hidden_list1=[]
    hidden_list2=[]
    with h5py.File(name, 'r') as f:
        # 访问 'data' 组
        data_group = f['data']
        # 遍历所有条目
        for entry_name in data_group:
            entry = data_group[entry_name]
            for key in entry:
                # 获取数据集的内容
                data = entry[key][()]
                # print(f"Entry: {entry_name}, Key: {key}")
                
            # exit()
            hidden_state_img1 =torch.tensor(entry['result_hidden_states_1'][()],dtype=torch.bfloat16)
            # hidden_state_img2 = torch.tensor(entry['result_hidden_states_2'][()],dtype=torch.bfloat16)
            hidden_list1.append(hidden_state_img1)
            # hidden_list2.append(hidden_state_img2)
    return hidden_list1

if __name__ == "__main__":
    # input the file path
    file_single=""  #single inference
    file_instruction="" #icc inference
    sample_num=500
    hidden_list_single = read_h5_single_(name=file_single)
    single_hidden = torch.stack(hidden_list_single).to(torch.float32)[:, 1:, -2, :]
    single_hidden_np_all_ = single_hidden.numpy()  # [sample_num, 756]
    single_hidden_np_all = single_hidden_np_all_[:sample_num]
    hidden_list1_inst,hidden_list2_inst=read_h5_(name=file_instruction)
    img_hidden_inst=torch.stack(hidden_list1_inst).to(torch.float32)[:,1:,-2,:]
    text_hidden_inst=torch.stack(hidden_list2_inst).to(torch.float32)[:,1:,-2,:]
    img_hidden_np_all_inst_ = img_hidden_inst.numpy()  # [sample_num, 756]
    text_hidden_np_all_inst_ = text_hidden_inst.numpy()  # [sample_num, 756]
    sample_num_inst=100
    img_hidden_np_all_inst=img_hidden_np_all_inst_[:sample_num_inst]
    text_hidden_np_all_inst=text_hidden_np_all_inst_[:sample_num_inst]
    single_hidden_np_all_inst=single_hidden_np_all_[:sample_num_inst]
    j=20
    img_hidden_np_inst=img_hidden_np_all_inst[:,j,:]
    text_hidden_np_inst=text_hidden_np_all_inst[:,j,:]
    single_hidden_np_inst=single_hidden_np_all_inst[:,j,:]
        

    combined_hidden_inst=np.vstack([img_hidden_np_inst, text_hidden_np_inst, single_hidden_np_inst])
    pca = PCA(n_components=2)

    pca_result_inst=pca.fit_transform(combined_hidden_inst)

        
    c_img_inst = np.mean(pca_result_inst[:sample_num_inst], axis=0)
    c_text_inst = np.mean(pca_result_inst[sample_num_inst:2*sample_num_inst], axis=0)
    c_single_inst = np.mean(pca_result_inst[2*sample_num_inst:], axis=0)
        
    delta_c_img_text_inst = c_img_inst - c_text_inst
    delta_c_img_single_inst = c_img_inst - c_single_inst
    delta_c_text_single_inst = c_text_inst - c_single_inst

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result_inst[:sample_num_inst, 0], pca_result_inst[:sample_num_inst, 1], c='blue',label='Img-Prefer', alpha=0.6, s=15)
    plt.scatter(pca_result_inst[sample_num_inst:2*sample_num_inst, 0], pca_result_inst[sample_num_inst:2*sample_num_inst, 1],label='Text-Prefer', c='green', alpha=0.6, s=15)
    plt.scatter(pca_result_inst[2*sample_num_inst:, 0], pca_result_inst[2*sample_num_inst:, 1], c='gray',  label='Original', alpha=0.6, s=15)
    plt.scatter([c_img_inst[0]], [c_img_inst[1]], c='darkblue', marker='X', s=100)
    plt.scatter([c_text_inst[0]], [c_text_inst[1]], c='darkgreen', marker='X', s=100)
    plt.scatter([c_single_inst[0]], [c_single_inst[1]], c='black', marker='X', s=100)

    from matplotlib.patches import FancyArrowPatch

    ax = plt.gca()  


    arrow1 = FancyArrowPatch(posA=(c_text_inst[0], c_text_inst[1]),
                                posB=(c_text_inst[0] + delta_c_img_text_inst[0], c_text_inst[1] + delta_c_img_text_inst[1]),
                                arrowstyle='->', color='red', mutation_scale=25,linewidth=2)
    ax.add_patch(arrow1)



    arrow2 = FancyArrowPatch(posA=(c_single_inst[0], c_single_inst[1]),
                                posB=(c_single_inst[0] + delta_c_img_single_inst[0], c_single_inst[1] + delta_c_img_single_inst[1]),
                                arrowstyle='->', color='purple',  mutation_scale=10,linewidth=2)
    ax.add_patch(arrow2)

    arrow3 = FancyArrowPatch(posA=(c_single_inst[0], c_single_inst[1]),
                                posB=(c_single_inst[0] + delta_c_text_single_inst[0], c_single_inst[1] + delta_c_text_single_inst[1]),
                                arrowstyle='->', color='orange',  mutation_scale=20,linewidth=2)
    ax.add_patch(arrow3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
    plt.close()
      