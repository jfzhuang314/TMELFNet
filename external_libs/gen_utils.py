import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from sklearn.neighbors import KDTree
import json
from external_libs.pointops.functions import pointops
import trimesh

def np_to_pcd(arr, color=[1,0,0]):
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def save_pcd(path, arr):
    o3d.io.write_point_cloud(path, arr)


def np_to_mesh(vertices, faces, color=[0.5, 0.5, 0.5]):
    """
    将顶点和面数据转换为Open3D网格
    Args:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面索引数组，形状为 (M, 3)
        color: 网格颜色，默认为灰色
    """
    vertices = np.array(vertices)
    faces = np.array(faces.T)  # 要求是#F x 3

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    # 计算法向量用于更好的显示效果
    mesh.compute_vertex_normals()
    # 设置颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector([color] * len(mesh.vertices))

    return mesh


def save_mesh(path, mesh):
    o3d.io.write_triangle_mesh(path, mesh)

def count_unique_by_row(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(a)
    np.put(b, ind, cnt)
    return b

def load_mesh(mesh_path, only_tooth_crop = False):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    if only_tooth_crop:
        cluster_idxes, cluster_nums, _ = mesh.cluster_connected_triangles()
        cluster_idxes = np.asarray(cluster_idxes)
        cluster_nums = np.asarray(cluster_nums)
        tooth_cluster_num = np.argmax(cluster_nums)
        mesh.remove_triangles_by_mask(cluster_idxes!=tooth_cluster_num)
    return mesh

def get_colored_mesh(mesh, label_arr, palte_type='default'):
    default_palte = np.array([
        # [255,153,153],
        [128, 128, 128],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255

    label15_palte = np.array([
    [ 0.9, 0.9, 0.9 ],      # 0
    [ 0.6, 0.3, 0.0 ],      # 1
    [ 0.6, 0.6, 0.0 ],      # 2
    [ 0.3, 0.6, 0.0 ],      # 3
    [ 0.0, 0.6, 0.6 ],      # 4
    [ 0.0, 0.0, 0.6 ],      # 5
    [ 0.6, 0.0, 0.6 ],      # 6
    [ 0.6, 0.0, 0.3 ],      # 7
    [ 0.5, 0.25, 0.0 ],     # 8
    [ 0.3, 0.3, 0.0 ],    # 9
    [ 0.15, 0.3, 0.0 ],     # 10
    [ 0.0, 0.3, 0.3 ],      # 11
    [ 0.0, 0.0, 0.3 ],      # 12
    [ 0.3, 0.0, 0.3 ],      # 13
    [ 0.3, 0.0, 0.15 ]         # 14
    ])

    palte = label15_palte if palte_type=='label15' else default_palte
    palte[9:] *= 0.4 if palte_type=='default' else 1.0

    label_arr = label_arr.copy()
    label_arr %= palte.shape[0]
    label_colors = np.zeros((label_arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[label_arr==idx] = palte[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    return mesh

def np_to_pcd_with_label(arr, label_arr=None, axis=3):
    if type(label_arr) == np.ndarray:
        arr = np.concatenate([arr[:,:3], label_arr.reshape(-1,1)],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    arr = arr.copy()
    arr[:,axis] %= palte.shape[0]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def get_number_from_name(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def get_up_from_name(path):
    return os.path.basename(path).split("_")[-1].split(".")[0]=="up"

def resample_pcd(pcd_ls, n, method):
    """Drop or duplicate points so that pcd has exactly n points"""
    if method=="uniformly":
        idx = np.random.permutation(pcd_ls[0].shape[0])
    elif method == "fps":
        idx = fps(pcd_ls[0][:,:3], n)
    pcd_resampled_ls = []
    for i in range(len(pcd_ls)):
        pcd_resampled_ls.append(pcd_ls[i][idx[:n]])
    return pcd_resampled_ls

def fps(xyz, npoint):
    if xyz.shape[0]<=npoint:
        raise "new fps error"
    xyz = torch.from_numpy(np.array(xyz)).type(torch.float).cuda()
    idx = pointops.furthestsampling(xyz, torch.tensor([xyz.shape[0]]).cuda().type(torch.int), torch.tensor([npoint]).cuda().type(torch.int)) 
    return torch_to_numpy(idx).reshape(-1)

def ba_fps(xyz, npoint, init_ratio=0.8, weights=None):
    """
    结合 FPS 和重要性采样的混合采样方法, importance sampling is
    Args:
        xyz: 点坐标数组 [N, 3]
        npoint: 目标采样点数
        init_ratio: 第一轮 FPS 采样保留的比例 (0~1)，默认 0.8
                    FPS 采样点数 = npoint * init_ratio
        weights: 每个顶点的权重 [N], None 则使用均匀权重
    Returns:
        selected_idx: 采样点的索引 [npoint]
    Example:
        xyz = np.random.rand(10000, 3)
        weights = np.random.rand(10000)  # 随机权重
        idx,_,_ = IS_fps(xyz, npoint=2000, init_ratio=0.7, weights=weights)
        print(len(idx))  # 2000
    """
    N = xyz.shape[0]
    # ========== 参数验证 ==========
    if npoint >= N:
        return np.arange(N)

    if not (0 < init_ratio <= 1):
        raise ValueError(f"init_ratio 必须在 (0, 1] 范围内，当前为 {init_ratio}")

    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.array(weights).reshape(-1)
        if weights.shape[0] != N:
            raise ValueError(f"weights 长度 ({weights.shape[0]}) 必须与点数 ({N}) 一致")
        # ========== 【关键】验证 weights 非负 ==========
        if np.any(weights < 0):
            negative_count = np.sum(weights < 0)
            min_weight = weights.min()
            raise ValueError(
                f"weights 必须全部 >= 0，但发现 {negative_count} 个负值\n"
                f"最小权重值：{min_weight}\n"
                f"请确保所有权重为非负数"
            )

    # ========== 步骤 1: 计算 FPS 和 IS 的采样数 ==========
    n_fps = int(npoint * init_ratio)
    n_is = npoint - n_fps

    # 确保至少采样 1 个 FPS 点
    n_fps = max(1, n_fps)

    # ========== 步骤 2: 执行 FPS 采样 ==========
    # FPS 采样前 n_fps 个点
    fps_idx = fps(xyz, n_fps)  # [n_fps]

    # ========== 步骤 3: 将已选中的 FPS 点权重置为 0 ==========
    weights_copy = weights.copy()
    weights_copy[fps_idx] = 0

    # ========== 步骤 4: 归一化权重 ==========
    weight_sum = weights_copy.sum()
    if weight_sum > 0:
        probs = weights_copy / weight_sum
    else:
        # 如果所有权重都为 0（理论上不应该），使用均匀分布
        probs = np.ones(N)
        probs[fps_idx] = 0
        probs = probs / probs.sum()

    # ========== 步骤 5: 重要性采样剩余的点 ==========
    if n_is > 0:
        # 使用 numpy 的 choice 进行有放回/无放回采样
        # replace=False 确保不重复选择已选中的点
        is_idx = np.random.choice(N,size=n_is,replace=False,p=probs)  # [n_is]
        # ========== 步骤 6: 合并 FPS 和 IS 的结果 ==========
        all_idx = np.concatenate([fps_idx, is_idx])
    else:
        all_idx = fps_idx
        is_idx = None

    return all_idx


def print_3d(*data_3d_ls):
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)
    o3d.visualization.draw_geometries(data_3d_ls,
            mesh_show_wireframe = True, mesh_show_back_face = True)

def print_3d_args(*data_3d_ls, mesh_show_wireframe = False, mesh_show_back_face = False,save_path=None):
    # 将传入的参数转换为列表
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)

    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = mesh_show_wireframe
    opt.mesh_show_back_face = mesh_show_back_face

    # 添加几何对象
    for geom in data_3d_ls:
        vis.add_geometry(geom)

    # 进行渲染
    vis.poll_events()
    vis.update_renderer()

    # 如果指定了保存路径，则保存图片
    if save_path:
        # 获取保存路径的目录部分
        save_dir = os.path.dirname(save_path)
        # 如果目录不存在，则创建目录，exist_ok=True 表示若目录已存在不会报错
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        vis.capture_screen_image(save_path)

    # 关闭窗口
    vis.destroy_window()



def print_3d_args_hdpi(*data_3d_ls, mesh_show_wireframe = False, mesh_show_back_face = False,
                       save_path=None, resolution=(3840, 2160)):
    # 将传入的参数转换为列表
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)

    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = mesh_show_wireframe
    opt.mesh_show_back_face = mesh_show_back_face

    # ctr = vis.get_view_control()
    # ctr.set_zoom(2.0)  # 应用最优缩放

    # 添加几何对象
    for geom in data_3d_ls:
        vis.add_geometry(geom)

    # 进行渲染
    vis.poll_events()
    vis.update_renderer()

    # 如果指定了保存路径，则保存图片
    if save_path:
        # 获取保存路径的目录部分
        save_dir = os.path.dirname(save_path)
        # 如果目录不存在，则创建目录，exist_ok=True 表示若目录已存在不会报错
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)


        image = vis.capture_screen_float_buffer(do_render=True)
        img_np = np.asarray(image)
        # 转换为 PIL Image
        pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
        # 计算实际 DPI（假设打印尺寸为 8x8 英寸）

        width, height = resolution
        # DPI = 像素数 / 英寸数
        print_size_inches = 8  # 假设打印为 8 英寸
        actual_dpi = width / print_size_inches

        # 保存并设置 DPI 信息
        pil_img.save(save_path, dpi=(actual_dpi, actual_dpi), quality=95)

        print(f"✅ 保存成功：{save_path}")
        print(f"   分辨率：{width}x{height} 像素")
        print(f"   DPI: {actual_dpi:.0f} (打印尺寸：{print_size_inches}x{print_size_inches} 英寸)")

        return actual_dpi


    # 关闭窗口
    vis.destroy_window()


def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)

def load_np(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr

def axis_rotation(axis, angle):
    ang = np.radians(angle) 
    R=np.zeros((3,3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang)+ux*ux*(1-cos(ang))
    R[0][1] = ux*uy*(1-cos(ang)) - uz*sin(ang)
    R[0][2] = ux*uz*(1-cos(ang)) + uy*sin(ang)
    R[1][0] = uy*ux*(1-cos(ang)) + uz*sin(ang)
    R[1][1] = cos(ang) + uy*uy*(1-cos(ang))
    R[1][2] = uy*uz*(1-cos(ang))-ux*sin(ang)
    R[2][0] = uz*ux*(1-cos(ang))-uy*sin(ang)
    R[2][1] = uz*uy*(1-cos(ang))+ux*sin(ang)
    R[2][2] = cos(ang) + uz*uz*(1-cos(ang))
    return R

def make_coord_frame(size=1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)

def save_json(file_path, json_obj):
    with open(file_path, "w") as json_file:

        json.dump(json_obj, json_file)

def read_txt(file_path):
    f = open(file_path, 'r')
    path_ls = []
    while True:
        line = f.readline().split()
        if not line: break
        path_ls.append(os.path.join(os.path.dirname(file_path), line.split("\n")[0] + ".npy"))
    f.close()

    return path_ls

def read_txt_obj_ls(path, ret_mesh=False, use_tri_mesh=False):
    # In some cases, trimesh can change vertex order
    if use_tri_mesh:
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1
    else:
        f = open(path, 'r')
        vertex_ls = []
        tri_ls = []
        #vertex_color_ls = []
        while True:
            line = f.readline().split()
            if not line: break
            if line[0]=='v':
                vertex_ls.append(list(map(float,line[1:4])))
                #vertex_color_ls.append(list(map(float,line[4:7])))
            elif line[0]=='f':
                tri_verts_idxes = list(map(str,line[1:4]))
                if "//" in tri_verts_idxes[0]:
                    for i in range(len(tri_verts_idxes)):
                        tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
                tri_verts_idxes = list(map(int, tri_verts_idxes))
                tri_ls.append(tri_verts_idxes)
            else:
                continue
        f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)

    vertex_ls = np.array(vertex_ls)
    output = [np.concatenate([vertex_ls,norms], axis=1)]

    if ret_mesh:


        output.append(mesh)
    return output


#---------------------------自定义类---------------------------#
