import argparse
import os
import numpy as np
from glob import glob

import torch
from torch_geometric.data import Data

import datasets.preprocess.utils.gen_utils as gu

parser = argparse.ArgumentParser()
parser.add_argument('--source_obj_data_path', default=r"E:\mesh_datasets\3DTeethChallenge\data_obj_parent_directory", type=str, help="data path in which original .obj data are saved")
parser.add_argument('--source_json_data_path', default=r"E:\mesh_datasets\3DTeethChallenge\data_json_parent_directory", type=str, help="data path in which original .json data are saved")
parser.add_argument('--save_data_path', default=r"E:\mesh_datasets\3DTeeth\data\DBGAMesh\raw", type=str, help="data path in which processed data will be saved")
args = parser.parse_args()

"""
propomts：
1. mesh是一个字典对象，包含了V,F,vLabel三个字段；
2. 要求将其转换为torch格式；
3. 保存为pt文件；保存参考代码为：torch.save(processed_data, processed_path)
"""


SAVE_PATH = args.save_data_path
SOURCE_OBJ_PATH = args.source_obj_data_path
SOURCE_JSON_PATH = args.source_json_data_path
Y_AXIS_MAX = 33.15232091532151
Y_AXIS_MIN = -36.9843781139949

os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

stl_path_ls = []
for dir_path in [
    x[0] for x in os.walk(SOURCE_OBJ_PATH)
    ][1:]:
    stl_path_ls += glob(os.path.join(dir_path,"*.obj"))

json_path_map = {}
for dir_path in [
    x[0] for x in os.walk(SOURCE_JSON_PATH)
    ][1:]:
    for json_path in glob(os.path.join(dir_path,"*.json")):
        json_path_map[os.path.basename(json_path).split(".")[0]] = json_path

all_labels = []
for i in range(len(stl_path_ls)):
    print(i, end=" ")
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
    loaded_json = gu.load_json(json_path_map[base_name])
    labels = np.array(loaded_json['labels']).reshape(-1,1)
    if loaded_json['jaw'] == 'lower':
        labels -= 20
    labels[labels//10==1] %= 10
    labels[labels//10==2] = (labels[labels//10==2]%10) + 8
    labels[labels<0] = 0
        
    vertices, org_mesh = gu.read_txt_obj_ls(stl_path_ls[i], ret_mesh=True, use_tri_mesh=False)

    vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
    #vertices[:, :3] = ((vertices[:, :3]-vertices[:, 1].min())/(vertices[:, 1].max() - vertices[:, 1].min()))*2-1
    vertices[:, :3] = ((vertices[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX - Y_AXIS_MIN))*2-1

    labeled_vertices = np.concatenate([vertices,labels], axis=1)

    name_id = str(base_name)

    # 从 org_mesh 提取面片 F
    F = np.asarray(org_mesh.triangles)

    # 转换为 torch 格式
    processed_data = {
        'V': torch.from_numpy(vertices[:,:3]).float(),
        'F': torch.from_numpy(F).long(),
        'vLabel': torch.from_numpy(labels).long(),
        'filename': name_id
    }

    # 保存为 pt 文件
    processed_path = os.path.join(SAVE_PATH, f"{name_id}.pt")
    torch.save(processed_data, processed_path)

