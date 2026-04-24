
import os
import json
import glob
import torch
from collections import namedtuple
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from external_libs import ops_utils as ou, gen_utils as gu

def topo_cluster(json_obj, opts):
    """
        根据输入的JSON对象和参数选项，边缘增强处理并裁剪牙齿面片。
        参数:
        json_obj (namedtuple): 包含点云、面、标签等信息的命名元组。
        opts (dict): 包含处理参数的字典，如alpha形状参数、最大迭代次数等。
        返回:
        tuple: 包含聚类数量、聚类中心、聚类点数量、聚类类标到真实类标的映射、
               最近邻裁剪索引、聚类真实标签、聚类真实二值标签和牙齿标志的元组。
        clu_num, clu_cen, clu_counts, clu_gt_map, nn_crop_indexes, clu_gt_label, clu_gt_bin_ls, tooth_flag
        """
    # 在极少数情况下可能会失效
    try:
        ette_lf = compute_GTTELF(json_obj.V, json_obj.F, json_obj.edge_pd_2, json_obj.edge_pd_3, json_obj.bg_pd_2, opts)
    except Exception as e:
        print(f"An error occurred: {e}")
        ette_lf = json_obj.edge_pd_3

    # enhanced tooth-all edge logits field
    etae_lf = np.squeeze(json_obj.edge_pd_2) + np.squeeze(ette_lf)
    tg_lf = np.squeeze(json_obj.bg_pd_2)

    # A. 获取各类mask
    raw_etae_mask = etae_lf > 0
    tg_mask = tg_lf > 0  # tooth-gin edge mask

    # print(json_obj.json_name)
    etae_mask = (tg_mask & raw_etae_mask).astype(float) > 0
    # sep_teeth_mask = get_tooth_flag_with_faces(etae_mask, json_obj)

    # 提取无边缘的牙齿点
    sep_teeth_mask = np.zeros_like(tg_lf, dtype=bool)  # sep-teeth mask；tae_mask，tg_mask,tte_mask的形状是相同的
    tooth_faces = get_filtered_faces(tg_mask, json_obj.F)  # tooth faces
    etae_faces = get_filtered_faces(etae_mask, json_obj.F)  # tae faces
    sep_teeth_faces = setdiff(tooth_faces, etae_faces)  # sep-teeth faces
    sep_teeth_mask[np.unique(sep_teeth_faces)] = True  # sep_teeth_mask=tooth_mask-tae_mask；实验表明根据面片进行提取的效果会更好些

    # B. 获取连通标签
    labels_topy = get_conn_labels(json_obj.V, json_obj.F, sep_teeth_mask)
    # C. DBSCAN（也可以在上述连通量上直接过滤，此处是为了方便获取聚类结果）
    xyz = json_obj.V[sep_teeth_mask, :]  # 获取sep-teeth的点
    xyzl = np.hstack([xyz, labels_topy[sep_teeth_mask].reshape(-1, 1) * 100])  # xyz+label
    db = DBSCAN(eps=0.2, min_samples=15).fit(xyzl)  # 借助dbscan获取聚类结果
    cluster_labels = db.labels_ # [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]
    gt_labels_tooth = json_obj.gt_labels[sep_teeth_mask, :]

    # 整个模型的聚类标签
    clu_labels_all = -1*np.ones_like(tg_lf, dtype=int)
    clu_labels_all[sep_teeth_mask] = cluster_labels
    # clu_labels_all += 1 # [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,14]

    # 收集聚类信息
    num_classes = opts['num_classes']
    unique_classes, cls_counts = np.unique(json_obj.gt_labels, return_counts=True)
    gt_cls_counts = np.zeros(num_classes, dtype=int)
    gt_cls_counts[unique_classes] = cls_counts
    clu_num, clu_cen, clu_counts, clu_gt_map, counts_mat, gt_clu_num_prop = get_clusters(cluster_labels,
                                                                                         gt_labels_tooth, xyz,
                                                                                         gt_cls_counts, num_classes)

    return (clu_num, clu_cen, clu_counts, clu_gt_map, counts_mat, gt_clu_num_prop,
            sep_teeth_mask,clu_labels_all)

    # clu_gt_bin_ls, clu_gt_label, nn_crop_indexes = get_patches(clu_cen, clu_gt_map, V,gt_labels, opts)
    # return (clu_num, clu_cen, clu_counts, clu_gt_map, nn_crop_indexes, clu_gt_label,
    #         clu_gt_bin_ls, sep_teeth_mask)


#region eef
def compute_GTTELF(V, F, tae_lf, tte_lf, tg_lf, opts):
    """
    ETTELF: enhanced tooth-tooth edge logits field
    处理牙齿轮廓数据，提取边缘点和牙齿点，并进行轮廓优化
    参数:
        V: 点云数据，形状为(n_points, 3)
        F: 面数据，形状为(n_faces, 3)
        tae_lf: tooth-all logits fields
        tte_lf: tooth-tooth logits fields
        tg_lf: tooth-gin logits fields
        opts: 包含处理参数的字典，应包含:
            alpha: alpha形状参数
            max_iter: 最大迭代次数
            iter_thred_len: 迭代阈值长度
            field_bound: 场边界
            field_sigma: 场标准差

    返回:
        ette_lf: 处理后的边缘点数据,enhanced tooth-tooth edge logits field
        C2: 优化后的轮廓点，顺时针排序
        part0, part1: 轮廓的两部分
        q1, q2: 边缘点在两部分轮廓上的最近邻点
    """
    # 提取TG-edges, BT-edges, Tooth points这三类点
    tte_mask = np.squeeze(tte_lf) > 0   # tooth-tooth edge mask
    raw_tae_mask = np.squeeze(tae_lf) > 0   # tooth-all edge mask
    tg_mask  = np.squeeze(tg_lf) > 0    # tooth-gin edge mask
    tae_mask = np.array(tg_mask & raw_tae_mask, dtype=np.float64) > 0  # filtered tooth-all edge mask

    # 选择投影边缘点
    tae_points_2d = V[tae_mask, 0:2]
    tte_points_2d = V[tte_mask, 0:2]

    # 提取无边缘的牙齿点
    sep_teeth_mask = np.zeros_like(tae_mask, dtype=bool)  # sep-teeth mask；tae_mask，tg_mask,tte_mask的形状是相同的
    tooth_faces = get_filtered_faces(tg_mask, F)          # tooth faces
    tae_faces = get_filtered_faces(tae_mask, F)           # tae faces
    sep_teeth_faces = setdiff(tooth_faces, tae_faces)  # sep-teeth faces
    sep_teeth_mask[np.unique(sep_teeth_faces)] = True # sep_teeth_mask=tooth_mask-tae_mask；实验表明根据面片进行提取的效果会更好些

    # 1. alphaShape构建2D网格形状
    as_bnd_points = tae_points_2d
    vertices, triangles = build_alpha_shape(as_bnd_points, opts['alpha'])
    mesh = create_mesh_from_vertices_and_triangles(vertices, triangles)

    boundary_vertices, boundary_loops = extract_boundary_with_halfedge(mesh, visualize=False)
    boundary_contour = vertices[boundary_loops[0], :]  # boundary_contour是顺时针分布的
    boundary_contour_3d = np.column_stack((boundary_contour, np.zeros(boundary_contour.shape[0])))

    # 2. 迭代优化轮廓
    C0 = boundary_contour.copy()
    avg_len = 10e3  # 初始平均边长设置为10000

    for i in range(opts['max_iter']):
        if avg_len < opts['iter_thred_len']:
            print(f"迭代 {i + 1}: 平均边长 {avg_len:.2f} 小于阈值 {opts['iter_thred_len']}，停止迭代")
            break

        C = C0.copy()
        C2, org_flag, avg_len = interp_contour(C)  # 插值
        C2 = update_interp_points(C2, np.zeros_like(org_flag, dtype=bool), V, tae_mask) # 位置更新
        C0 = C2

    C0_3d = np.column_stack((C0, np.zeros(C0.shape[0])))

    gu.print_3d(mesh, boundary_contour_3d, C0_3d)


    # 3. 划分为上下两段, 提取轮廓的两部分
    segment_labels = label_contour_segment(C2)
    idx_0, idx_1 = (segment_labels == 0), (segment_labels == 1)
    part0, part1 = C2[idx_0], C2[idx_1]

    # 4. 为每个边缘点找到在两部分轮廓上的最近邻点
    tree0, tree1 = KDTree(part0), KDTree(part1)
    _, q1_idx = tree0.query(tte_points_2d, k=1)
    _, q2_idx = tree1.query(tte_points_2d, k=1)
    q1, q2 = part0[q1_idx], part1[q2_idx]

    # 5. 计算点到线段的距离并应用场效应
    V_2d = V[:, :2]
    distances = point_to_segment_distance(V_2d, sep_teeth_mask, q1.squeeze(), q2.squeeze())

    # 将大于field_bound的距离设置为10000
    distances[distances > opts['field_bound']] = 10e3

    # 计算场效应 (使用指数衰减模型)
    ette_lf = np.nansum(np.exp(-distances / opts['field_sigma']), axis=1)
    # new_edge_pd_2 = tae_lf.squeeze() + eef

    return ette_lf

def get_filtered_faces(vertex_flag, F):
    """过滤面片，只保留所有顶点都满足条件的面片,F必须是n3格式"""
    assert F.shape[1] == 3, "输入数组必须是n3格式"

    face_flag = np.all(vertex_flag[F], axis=1)
    filtered_F = F[face_flag, :]

    return filtered_F    # 过滤面片

def check_n3( x):
    """检查输入的数组是否满足n3条件，主要用以V及F"""
    x = x.T if x.shape[0] == 3 else x
    return x

def check_3n( x):
    """检查输入的数组是否满足3n格式，主要用以V及F"""
    x = x.T if x.shape[1] == 3 else x
    return x

def setdiff(a, b):
    """返回 a 中不在 b 中的元素，保持顺序
    a,b都要满足n3格式
    """
    assert a.shape[1] == 3 and b.shape[1] == 3, "输入数组必须是n3格式"

    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    a_rows_view = a.view([('', a.dtype)] * a.shape[1])
    b_rows_view = b.view([('', b.dtype)] * b.shape[1])
    diff = np.setdiff1d(a_rows_view, b_rows_view)
    return diff.view(a.dtype).reshape(-1, a.shape[1])

# 定义一个函数，用于收集聚类信息
def get_clusters(cluster_labels, gt_labels_tooth, sep_teeth_pts, cls_counts, num_classes=15):
    """
    :param cluster_labels: 聚类标签
    :param gt_labels_tooth: ground truth labels of teeth
    :param sep_teeth_pts: sep_teeth_points
    :return:
    num_clusters:聚类的数量, 1
    cluster_cen:每个聚类的中心点, [Clu,3]
    counts: 每个聚类的点数量, [Clu,1]
    cluster_gt_map: 聚类类标到ground truth类标的映射, [Clu,1]
    counts_mat: [15,Clu]，每行中的值代表该clu占据gt类的个数
    gt_clu_num_prop：[15,Clu]，每行中的值代表该clu占据gt类的比例，通常这个比例不能太小
    """
    unique_labels = np.unique(cluster_labels)

    # 聚类相关信息：聚类数量、聚类大小、聚类中心、聚类标签
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 排除噪声点(-1)
    cluster_cen = np.zeros((num_clusters, 3))
    counts = np.zeros(num_clusters)

    # 求取聚类中心点
    idx = 0
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue
        mask = (cluster_labels == label)
        cluster_cen[idx] = np.mean(sep_teeth_pts[mask], axis=0)
        counts[idx] = np.sum(mask)
        idx += 1

    # 聚类类标与真实类标的映射关系
    # valid_cluster_mask = (cluster_labels >= 0) & (gt_labels_tooth.squeeze() >= 1)
    valid_cluster_mask = (cluster_labels >= 0)
    valid_cluster_labels = cluster_labels[valid_cluster_mask].astype(int)
    valid_gt_labels = gt_labels_tooth[valid_cluster_mask].astype(int) # 可能包括背景
    row_indices = valid_gt_labels.squeeze()
    col_indices = valid_cluster_labels.squeeze()

    # 确保聚类个数的一致性
    counts_mat = np.zeros((num_classes,cluster_cen.shape[0])) #  15 * clu。与matlab中的混淆矩阵一致

    for i, _ in enumerate(row_indices):
        r, c = row_indices[i], col_indices[i]
        counts_mat[r, c] += 1

    # 找出每行最大值的索引位置，如果没有的话，则映射为0，表示是背景点
    clu_gt_label_map = np.argmax(counts_mat, axis=0) # clu对应的label
    # gt-clu的真实类别占比
    eps = 1e-8
    gt_clu_num_prop = counts_mat /(cls_counts.reshape(-1, 1)+eps)

    return (num_clusters, cluster_cen, counts,clu_gt_label_map, counts_mat, gt_clu_num_prop)

# def get_patches(clu_cen, clu_gt_map, V, gt_labels, clu_counts, opts):
def get_patches(pred_clu, V , gt_labels, opts):
    """
    :param clu_cen: 聚类中心, [Clu,3]
    :param clu_gt_map: 聚类类标到ground truth类标的映射, [Clu,1]
    :param V: 点云, [N,3]
    :param gt_labels: 点云的ground truth类标, [N,1]
    :param opts: 超参数, 包含crop_num
    :return:
    """
    clu_labels_all = np.array(pred_clu['clu_labels_all'])  # clu, 3072
    clu_num = np.array(pred_clu['clu_num']) # 1
    clu_cen = np.array(pred_clu['clu_cen'])  # clu, 3
    clu_counts = np.array(pred_clu['clu_counts'])  # clu, 1
    clu_gt_map = np.array(pred_clu['clu_gt_map'])  # clu, 1
    counts_mat = np.array(pred_clu['counts_mat'])# clu, 1500
    gt_clu_num_prop = np.array(pred_clu['gt_clu_num_prop'])  # clu, 1
    sep_teeth_mask = np.array(pred_clu['sep_teeth_mask']) # 10000,

    crop_num = opts['crop_num']
    org_xyz = np.expand_dims(V, axis=0)  # 1NC格式
    sampled_clusters = np.expand_dims(clu_cen, axis=0)  # 1N3格式
    nn_crop_indexes = ou.get_nearest_neighbor_idx(org_xyz, sampled_clusters, crop_num)  # 1, Clu, 1500
    clu_gt_label = ou.get_indexed_features(np.expand_dims(gt_labels, axis=0).transpose(0, 2, 1),nn_crop_indexes)  # Clu,1,1500
    clu_gt_bin_ls = np.zeros(clu_gt_label.shape)  # [Clu,1,crop_num]
    for i in range(clu_gt_label.shape[0]):
        try:
            clu_gt_bin_ls[i, 0, clu_gt_map[i] == clu_gt_label[i][0]] = 1
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")
            print(clu_gt_map, nn_crop_indexes[0].shape[0], clu_gt_map.shape[0])
            pass

    # 计算 nn_crop_indexes 两两之间的重合率
    num_clusters = nn_crop_indexes[0].shape[0]
    overlap_ratio_mat = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        set_i = set(nn_crop_indexes[0][i])
        for j in range(i, num_clusters):
            set_j = set(nn_crop_indexes[0][j])
            intersection = len(set_i.intersection(set_j))
            ratio = intersection / crop_num
            overlap_ratio_mat[i][j] = ratio
            overlap_ratio_mat[j][i] = ratio  # 重合率矩阵是对称的

    # 对于overlap_ratio_mat，利用NMS进行筛选，如果重合率大于0.85的话，则将clu_counts比较少的删掉
    # NMS筛选
    keep_indices = []
    removed_indices = []  # 新增：存储被删除的索引

    # 按counts降序排序
    sorted_indices = sorted(range(num_clusters), key=lambda x: -clu_counts[x])
    for i in sorted_indices:
        keep = True
        for j in keep_indices:
            if overlap_ratio_mat[i][j] > 0.9:  # 重合率阈值
                keep = False
                removed_indices.append(i)  # 记录被删除的索引
                break
        if keep:
            keep_indices.append(i)

    # # 打印被删除的索引及其相关信息
    # if len(removed_indices) > 0:
    #     print(f"被删除的聚类索引: {removed_indices}")
    #     for idx in removed_indices:
    #         print(f"索引 {idx}: counts={clu_counts[idx]}, 与保留聚类的最大重合率={max(overlap_ratio_mat[idx, keep_indices], default=0)}")

    # 只保留筛选后的聚类
    nn_crop_indexes[0] = nn_crop_indexes[0][keep_indices]
    clu_gt_bin_ls = clu_gt_bin_ls[keep_indices]
    clu_gt_label = clu_gt_label[keep_indices]
    overlap_ratio_mat = overlap_ratio_mat[keep_indices,:]
    overlap_ratio_mat = overlap_ratio_mat[:,keep_indices]

    pred_clu['clu_labels_all'] = clu_labels_all[keep_indices]
    pred_clu['clu_num'] = len(keep_indices)
    pred_clu['clu_cen'] = pred_clu['clu_cen'][keep_indices]
    pred_clu['clu_counts'] = pred_clu['clu_counts'][keep_indices]
    pred_clu['clu_gt_map'] = pred_clu['clu_gt_map'][keep_indices]
    pred_clu['counts_mat'] = pred_clu['counts_mat'][:,keep_indices]
    pred_clu['gt_clu_num_prop'] = pred_clu['gt_clu_num_prop'][:,keep_indices]

    pred_clu['nn_crop_indexes'] = nn_crop_indexes
    pred_clu['clu_gt_bin_ls'] = clu_gt_bin_ls
    pred_clu['clu_gt_label'] = clu_gt_label
    pred_clu['overlap_ratio_mat'] = overlap_ratio_mat


    return pred_clu


#endregion eef


def read_json_files(results_path):
    # 获取所有JSON文件路径
    json_files = glob.glob(os.path.join(results_path, '*.json'))
    json_data_list = []
    pred_accs = []
    # 遍历每个JSON文件
    for file_path in json_files:
        try:
            # 读取JSON文件内容
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # 处理JSON数据，与MATLAB代码保持一致
            # 假设mesh_file是嵌套列表，提取第一个元素
            if isinstance(json_data.get('mesh_file'), list) and len(json_data['mesh_file']) > 0:
                json_data['mesh_file'] = json_data['mesh_file'][0][0]

            # 添加JSON文件名
            json_data['json_name'] = os.path.basename(file_path)

            # 存储处理后的JSON数据
            json_data_list.append(json_data)

            # 提取预测准确率
            if 'predict_acc' in json_data:
                pred_accs.append(json_data['predict_acc'])

            print(f'成功读取文件: {file_path}')

        except Exception as e:
            print(f'读取文件 {file_path} 时出错: {str(e)}')

    return json_data_list, pred_accs


def get_json_obj(json_data_list, i):
    """
    从JSON数据列表中获取特定索引的对象，并转换为结构化格式
    参数:
        json_data_list: JSON数据列表
        i: 要获取的对象索引
    返回:
        json_obj: 结构化的JSON对象
    """
    # 确保索引在有效范围内
    if i < 0 or i >= len(json_data_list):
        raise IndexError(f"索引 {i} 超出范围 [0, {len(json_data_list) - 1}]")

    # 获取指定索引的JSON数据
    json_data = json_data_list[i]

    # 创建一个命名元组作为结构体
    JsonObj = namedtuple('JsonObj', [
        'json_name',
        'points',# 点云输入特征
        'V',  # 点云坐标
        'F',  # 面信息
        'gt_labels',  # 真实标签
        'pred_labels',  # 预测标签
        'bg_pd_1',  # 背景属性1
        'bg_weight_1',  # 背景权重1
        'bg_pd_2',  # 背景属性2
        'edge_pd_1',  # 边缘属性1
        'edge_weight_1',  # 边缘权重1
        'edge_pd_2',  # 边缘属性2
        'edge_pd_3',  # 边缘属性3
        'edge_gt_bin_labels'  # 边缘二值标签
    ])

    # 提取并转换数据
    json_obj = JsonObj(
        json_name=json_data['json_name'],
        points=np.array(json_data['pts_normals'])[:, 0:6],
        V=np.array(json_data['pts_normals'])[:, 0:3],
        F=np.array(json_data['faces']),
        gt_labels=np.array(json_data['gt_labels']),
        pred_labels=np.array(json_data['pred_labels']),
        bg_pd_1=np.array(json_data['bg_pd_1']),
        bg_weight_1=np.array(json_data['bg_weight_1']),
        bg_pd_2=np.array(json_data['bg_pd_2']),
        edge_pd_1=np.array(json_data['edge_pd_1']),
        edge_weight_1=np.array(json_data['edge_weight_1']),
        edge_pd_2=np.array(json_data['edge_pd_2']),
        edge_pd_3=np.array(json_data['edge_wobg_pd_2']),
        edge_gt_bin_labels=np.array(json_data['edge_gt_bin_labels'])[:, np.newaxis]
    )

    return json_obj

#region functions
# alpha shape
def build_alpha_shape(points, alpha):
    """
    计算 alpha shape 的三角形面片
    参数:
    points: 点集，形状为 (n, 2)
    alpha: alpha 参数
    返回:
    vertices: 顶点坐标
    triangles: 三角形面片索引
    """
    # 进行 Delaunay 三角剖分
    tri = Delaunay(points)
    triangles = tri.simplices

    # 计算每个三角形的外接圆半径
    a = np.sqrt(np.sum((points[triangles[:, 0]] - points[triangles[:, 1]]) ** 2, axis=1))
    b = np.sqrt(np.sum((points[triangles[:, 1]] - points[triangles[:, 2]]) ** 2, axis=1))
    c = np.sqrt(np.sum((points[triangles[:, 2]] - points[triangles[:, 0]]) ** 2, axis=1))
    s = (a + b + c) / 2.0
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = a * b * c / (4.0 * areas + 10e-6)

    # 过滤掉外接圆半径大于 alpha 的三角形
    filtered_triangles = triangles[circum_r < alpha]

    return points, filtered_triangles


# 使用 Open3D 从顶点和三角形创建网格
def create_mesh_from_vertices_and_triangles(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    if vertices.shape[1] == 2:
        # 将 2D 坐标转换为 3D 坐标（z=0）
        vertices_3d = np.column_stack((vertices, np.zeros(vertices.shape[0])))
    else:
        vertices_3d = vertices

    # 设置顶点和三角形
    mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 计算顶点法线，使网格在可视化时正确显示光照
    mesh.compute_vertex_normals()

    # 可选：设置网格颜色
    mesh.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    # mesh.visualization.show_wireframe = True # 显示线框
    return mesh


# 提取网格的边界边并形成顺时针有序轮廓
def ensure_clockwise(boundary_loop, vertices):
    """
    确保边界环按顺时针排列
    参数:
    boundary_loop: 边界环的顶点索引列表
    vertices: 网格的所有顶点坐标
    返回:
    clockwise_loop: 顺时针排列的边界环
    """
    # 获取边界环的顶点坐标
    loop_vertices = vertices[boundary_loop]
    # 计算边界环的质心
    centroid = np.mean(loop_vertices, axis=0)
    # 选择三个连续的点计算叉积
    # 为了提高鲁棒性，我们选择相隔较远的点
    n = len(boundary_loop)
    if n < 3:
        # 如果环上的点少于3个，无法确定方向，直接返回
        return boundary_loop
    # 选择点0、点n//3和点2n//3
    p1 = loop_vertices[0]
    p2 = loop_vertices[n // 3]
    p3 = loop_vertices[2 * n // 3]
    # 计算向量v1 = p2 - p1和v2 = p3 - p2
    v1 = p2 - p1
    v2 = p3 - p2
    # 计算叉积
    cross_product = np.cross(v1[0:3], v2[0:3])
    # print(v1, v2, cross_product)
    # 计算从质心到p1的向量
    # centroid_to_p1 = p1 - centroid
    # 计算叉积与质心到p1向量的点积
    # dot_product = np.dot(cross_product, centroid_to_p1)
    dot_product = cross_product[-1]
    # print(dot_product)
    # 如果点积为正，则边界环是逆时针的，需要反转
    if dot_product < 0:
        return boundary_loop[::-1]  # 反转边界环
    else:
        return boundary_loop  # 边界环已经是顺时针的


def extract_boundary_with_halfedge(mesh, visualize=True):
    """
    使用HalfEdgeTriangleMesh的get_boundaries函数提取网格的边界，并确保边界环按顺时针排列
    参数:
    mesh: Open3D的TriangleMesh对象
    visualize: 是否可视化结果
    返回:
    boundary_vertices: 边界点的索引列表
    boundary_loops: 边界环列表，每个环是顶点索引的列表（按顺时针排列）
    """
    # 转换为HalfEdgeTriangleMesh
    halfedge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    # 使用get_boundaries函数提取边界环
    boundary_loops = halfedge_mesh.get_boundaries()
    # 如果没有边界环，返回空列表
    if not boundary_loops:
        print("警告: 网格没有边界")
        return [], []

    # 获取网格的顶点坐标
    vertices = np.asarray(mesh.vertices)
    # 确保每个边界环按顺时针排列
    clockwise_loops = []
    for loop in boundary_loops:
        clockwise_loop = ensure_clockwise(loop, vertices)
        clockwise_loops.append(clockwise_loop)
    # 获取所有边界顶点
    boundary_vertices = set()
    for loop in clockwise_loops:
        for vertex_idx in loop:
            boundary_vertices.add(vertex_idx)
    boundary_vertices = list(boundary_vertices)

    # 可视化结果
    if visualize and clockwise_loops:
        # 创建原始网格的副本
        mesh_copy = o3d.geometry.TriangleMesh()
        mesh_copy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        mesh_copy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        mesh_copy.compute_vertex_normals()
        mesh_copy.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

        # 创建边界点的点云
        boundary_vertices_pcd = o3d.geometry.PointCloud()
        boundary_vertices_pcd.points = o3d.utility.Vector3dVector(
            vertices[boundary_vertices])
        boundary_vertices_pcd.paint_uniform_color([1, 0, 0])  # 红色

        # 创建边界环的线集
        line_sets = []

        # 为每个边界环创建一个线集
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1]]

        for i, loop in enumerate(clockwise_loops):
            # 创建线集
            line_set = o3d.geometry.LineSet()

            # 设置点
            loop_points = vertices[loop]
            line_set.points = o3d.utility.Vector3dVector(loop_points)

            # 设置线
            lines = []
            for j in range(len(loop) - 1):
                lines.append([j, j + 1])
            lines.append([len(loop) - 1, 0])  # 闭合环
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # 设置颜色
            color = colors[i % len(colors)]
            line_set.paint_uniform_color(color)

            line_sets.append(line_set)

        # 可视化网格和边界点
        o3d.visualization.draw_geometries(
            [mesh_copy, boundary_vertices_pcd],
            window_name="网格和边界点",
            mesh_show_wireframe=True
        )

        # 可视化网格和边界环
        o3d.visualization.draw_geometries(
            [mesh_copy] + line_sets,
            window_name="网格和边界环（顺时针）",
            mesh_show_wireframe=True
        )

    return boundary_vertices, clockwise_loops


def interp_contour(contour_points):
    """
    针对平均边长较长的边拆分为多条较短的边，要求contour_points是闭合的。
    参数:
    contour_points: 输入的轮廓点，形状为(n_points, 2)或(n_points, 3)
    返回:
    new_points: 新的点集，包含原始点和插值点
    is_original: 布尔数组，指示每个点是否为原始点
    avg_length: 原始轮廓的平均边长
    """
    n_points = contour_points.shape[0]
    if n_points < 3:
        raise ValueError('输入的轮廓点必须至少包含3个顶点')

    # 计算每条边的长度
    edge_lengths = np.zeros(n_points)
    for i in range(n_points):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % n_points]
        edge_lengths[i] = np.linalg.norm(p2 - p1)

    # 计算平均边长
    avg_length = np.mean(edge_lengths)
    # print(f'平均边长: {avg_length:.4f}')

    # 初始化结果列表
    new_points_list = []
    is_original_list = []

    # 处理每条边
    for i in range(n_points):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % n_points]
        edge_length = edge_lengths[i]

        # 添加原始起点
        new_points_list.append(p1)
        is_original_list.append(True)

        # 在长边上插入点
        if edge_length > avg_length:
            # 至少插入1个点
            n_insert = max(1, int(np.ceil(edge_length / avg_length)) - 1)
            for j in range(1, n_insert + 1):
                t = j / (n_insert + 1)
                new_point = p1 + t * (p2 - p1)
                new_points_list.append(new_point)
                is_original_list.append(False)

    # 转换为NumPy数组
    new_points = np.array(new_points_list)
    is_original = np.array(is_original_list, dtype=bool)

    return new_points, is_original, avg_length


def update_interp_points(new_points, is_original, V, flag_wo_bnd):
    """
    根据最近邻点的平均值更新插值点的位置
    参数:
    new_points: 包含原始点和插值点的点集，形状为(N, 2)或(N, 3)
    is_original: 布尔数组，指示每个点是否为原始点，形状为(N,)
    V: 参考点集，形状为(M, 2)或(M, 3)
    flag_wo_bnd: 布尔数组，指示V中哪些点参与计算，形状为(M,)

    返回:
    updated_points: 更新后的点集，形状为(N, 2)或(N, 3)
    """
    # 提取插值点
    interpolation_points = new_points[~is_original, :2]

    # 提取参与计算的参考点
    reference_points = V[flag_wo_bnd, :2]

    # 如果没有参考点，直接返回原结果
    if reference_points.size == 0:
        updated_points = new_points.copy()
        return updated_points

    # 对每个插值点，找到其在参考点中的3个最近邻点
    k = 3
    tree = KDTree(reference_points)
    distances, indices = tree.query(interpolation_points, k=k)

    # 初始化更新后的插值点
    updated_interpolation_points = np.zeros_like(interpolation_points)

    # 计算每个插值点的新位置
    for i in range(interpolation_points.shape[0]):
        # 获取3个最近邻点的坐标
        nearest_neighbors = reference_points[indices[i]]
        # 计算平均值作为新位置
        updated_interpolation_points[i] = np.mean(nearest_neighbors, axis=0)
    # 更新结果点集
    updated_points = new_points.copy()
    updated_points[~is_original, :2] = updated_interpolation_points
    return updated_points


def label_contour_segment(C):
    """
    标记顺时针有向轮廓中的特定线段
    参数:
    C: 顺时针有向的轮廓点集，形状为(N, 2)
    返回:
    segment_labels: 线段标记，形状为(N,)，0表示p1到p2的线段，1表示其余部分
    """
    # 寻找x<0区域的最高点p1
    left_points = C[C[:, 0] < 0]
    if len(left_points) == 0:
        raise ValueError("轮廓中没有x<0的点")
    idx_p1 = np.argmax(left_points[:, 1])
    p1 = left_points[idx_p1]

    # 寻找x>0区域的最高点p2
    right_points = C[C[:, 0] > 0]
    if len(right_points) == 0:
        raise ValueError("轮廓中没有x>0的点")
    idx_p2 = np.argmax(right_points[:, 1])
    p2 = right_points[idx_p2]

    # 在原始点集中找到p1和p2的索引
    # 使用KDTree实现knnsearch功能
    tree = KDTree(C)
    _, idx_p1_original = tree.query(p1.reshape(1, -1), k=1)
    _, idx_p2_original = tree.query(p2.reshape(1, -1), k=1)

    idx_p1_original = idx_p1_original.squeeze()
    idx_p2_original = idx_p2_original.squeeze()

    # 由于是顺时针方向，p1到p2的线段应该是从p1开始，沿着索引增加的方向到p2
    # 处理p1索引大于p2索引的情况（即线段跨越了点集的末尾和开头）
    N = len(C)
    segment_labels = np.ones(N, dtype=int)  # 初始全部标记为1

    if idx_p1_original < idx_p2_original:
        # p1在p2之前，直接标记p1到p2之间的线段为0
        segment_labels[idx_p1_original:idx_p2_original + 1] = 0
    else:
        # p1在p2之后，线段跨越了点集的末尾和开头
        segment_labels[idx_p1_original:] = 0
        segment_labels[:idx_p2_original + 1] = 0

    return segment_labels


def point_to_segment_distance(V, flag, header, tail):
    """
    计算点集到线段集的距离（完全向量化版本）

    参数:
    V: 点集，形状为(n, 2)或(n, 3)
    flag: 布尔数组，指示哪些点参与计算，形状为(n,)
    header: 线段起点集，形状为(m, 2)或(m, 3)
    tail: 线段终点集，形状为(m, 2)或(m, 3)
    返回:
    distances: 距离矩阵，形状为(n, m)，其中distances[i,j]表示点i到线段j的距离
    """
    n = V.shape[0]  # 顶点数量
    m = header.shape[0]  # 线段数量

    # 初始化距离矩阵
    distances = np.zeros((n, m))
    # 对flag为false的点，设置距离为一个很大的值
    distances[~flag, :] = 10e2
    # 获取flag为true的点的索引
    idxs = np.where(flag)[0]

    # 对每个顶点计算到所有线段的距离
    for i in idxs:
        p = V[i]
        # 计算线段向量
        ab = tail - header
        # 计算线段长度的平方
        ab_length_sq = np.sum(ab ** 2, axis=1) + 10e-6
        # 计算顶点到线段起点的向量
        ap = p - header
        # 计算投影比例 t
        t = np.sum(ap * ab, axis=1) / ab_length_sq
        # 根据投影比例确定距离计算方式
        # 预计算三种情况下的距离
        norm_ap = np.sqrt(np.sum(ap ** 2, axis=1))  # t < 0 时的距离
        bp = p - tail
        norm_bp = np.sqrt(np.sum(bp ** 2, axis=1))  # t > 1 时的距离
        proj = header + np.expand_dims(t, axis=1) * ab
        norm_p_minus_proj = np.sqrt(np.sum((p - proj) ** 2, axis=1))  # 0 <= t <= 1 时的距离

        # 创建标志向量
        flag_less_than_0 = t < 0
        flag_greater_than_1 = t > 1
        flag_between_0_1 = ~flag_less_than_0 & ~flag_greater_than_1

        # Ensure boolean masks are 1D
        flag_less_than_0 = flag_less_than_0.flatten()
        flag_greater_than_1 = flag_greater_than_1.flatten()
        flag_between_0_1 = flag_between_0_1.flatten()

        # Check if the lengths match
        assert len(flag_less_than_0) == m, f"flag_less_than_0 length {len(flag_less_than_0)} does not match m {m}"
        assert len(flag_greater_than_1) == m, f"flag_greater_than_1 length {len(flag_greater_than_1)} does not match m {m}"
        assert len(flag_between_0_1) == m, f"flag_between_0_1 length {len(flag_between_0_1)} does not match m {m}"

        # 根据标志向量选择合适的距离
        distances[i, flag_less_than_0] = norm_ap[flag_less_than_0]
        distances[i, flag_greater_than_1] = norm_bp[flag_greater_than_1]
        distances[i, flag_between_0_1] = norm_p_minus_proj[flag_between_0_1]

    return distances


# 定义一个函数，用于从标签获取RGB颜色（简化版，实际应根据需求调整）
def get_rgb_from_label15(teeth49, label):
    # 这里假设teeth49是一个参考数据，label是一个二进制标签
    # 实际应根据原始MATLAB函数实现
    colors = np.zeros((len(label), 3))
    colors[label > 0] = [1, 0, 0]  # 红色表示边缘
    colors[label <= 0] = [0.8, 0.8, 0.8]  # 灰色表示非边缘
    return colors



def get_conn_labels(vertices, F, vertex_flag):
    """
    计算网格的连通分量标签，并返回与vertex_flag相同形状的向量
    参数:
        vertices: 顶点坐标数组，形状为 (n_vertices, 3)
        F: 面片索引数组，形状为 (n_faces, 3)，假设为三角形网格
        vertex_flag: 布尔数组，表示顶点是否满足条件
    返回:
        labels: 与vertex_flag相同形状的向量，其中满足条件的顶点被标记为对应的连通分量编号
        components_info: 包含每个连通分量信息的列表，包括全局顶点索引和全局面片索引
    """
    # 对于每个面片，检查其所有顶点是否都满足条件
    face_flag = np.all(vertex_flag[F], axis=1)
    # 过滤面片
    filtered_F = F[face_flag]
    # 如果没有满足条件的面片，返回全零标签
    if len(filtered_F) == 0:
        return np.zeros_like(vertex_flag, dtype=int), []
    # 创建过滤后的网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=filtered_F)
    # 计算连通分量（不要求水密）
    components = mesh.split(only_watertight=False)
    # 初始化标签向量，不满足条件的顶点标记为0
    labels = np.zeros(len(vertices), dtype=int)
    # 为原始网格的顶点创建KD树，用于高效的最近邻查询
    kdtree = cKDTree(vertices)

    # 为每个连通分量的顶点分配唯一的标签编号
    for component_idx, component in enumerate(components, 1):  # 从1开始编号
        # 获取当前分量的所有顶点索引（局部索引，从0开始）
        component_vertices_local = np.unique(component.faces.flatten())
        # 使用批量KNN查询来找到原始网格中最接近的顶点
        if len(component_vertices_local) > 0:
            # 获取所有局部顶点的坐标
            component_vertices_coords = component.vertices[component_vertices_local]
            # 批量KNN查询：一次性查询所有局部顶点
            distances, global_indices = kdtree.query(component_vertices_coords, k=1)
            labels[global_indices] = component_idx

    return labels


#########################################################
# edge enhancement


# 使用 numpy 函数替代 MATLAB 的 setdiff 函数
# 注意：这是一个简化实现，可能需要根据具体需求调整

# ddf特征

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # Convert tensors to float32 if they are not already
    if src.dtype not in [torch.float32, torch.float64]:
        src = src.float()
    if dst.dtype not in [torch.float32, torch.float64]:
        dst = dst.float()

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)).float()
    dist += torch.sum(src ** 2, -1).view(B, N, 1).float()
    dist += torch.sum(dst ** 2, -1).view(B, 1, M).float()
    return dist

def get_ddf(cropped_coord, center_points):
    """
    :param self:
    :param cropped_coord: BNC格式
    :param center_points: B
    :return:
    """
    device = center_points.device
    cropped_coord = cropped_coord.to(device)

    B, N, C = cropped_coord.shape

    # Check if center_points is already a tensor
    if not isinstance(center_points, torch.Tensor):
        center_points = torch.from_numpy(center_points).cuda()
    # else:
    #     center_points = center_points.cuda()  # Move the tensor to GPU
    ddf = square_distance(cropped_coord, center_points)
    ddf = torch.sqrt(ddf)
    ddf *= (-4)
    ddf = torch.exp(ddf)
    ddf = ddf.permute(0, 2, 1)
    return ddf

#endregion functions


def get_tooth_flag_with_faces(edge_flag, json_obj):
    """
    原始牙齿面片-边缘面片=牙齿实例面片
    :param edge_flag:
    :param json_obj:
    :return:
    """
    tooth_flag = np.zeros_like(edge_flag, dtype=bool)
    org_tooth_faces = get_filtered_faces(np.squeeze(json_obj.bg_pd_2) > 0, json_obj.F)
    edges_faces = get_filtered_faces(edge_flag, json_obj.F)
    tooth_faces = setdiff(org_tooth_faces, edges_faces)
    tooth_flag[np.unique(tooth_faces)] = True
    return tooth_flag

def get_max_frequency_non_zero_class(input_vector):
    """
    提取向量中非零类标中频数最大的值
    参数:
        input_vector: 形状为(15, 1500)的二维向量
    返回:
        包含1500个元素的一维数组，每个元素是对应行中非零类标中频数最大的值
    """
    # 确保输入是numpy数组
    input_array = np.asarray(input_vector)

    # 转置输入数组，使我们可以按列处理
    transposed_array = input_array.T  # 现在形状为(1500, 15)
    # 初始化结果数组
    result = np.zeros(transposed_array.shape[0], dtype=input_array.dtype)
    # 对每一列进行处理（现在是转置后的每一行）
    for i in range(transposed_array.shape[0]):
        # 获取当前列（转置后的行）的非零元素
        non_zero_elements = transposed_array[i, transposed_array[i] != 0]
        # 如果没有非零元素，则结果为0
        if non_zero_elements.size == 0:
            result[i] = 0
        else:
            # 计算非零元素的唯一值和对应的计数
            values, counts = np.unique(non_zero_elements, return_counts=True)
            # 获取最大计数对应的索引
            max_count_idx = np.argmax(counts)
            # 将最大频数的非零元素存入结果
            result[i] = values[max_count_idx]
    return result


def vis_crop_bin_labels(json_obj, nn_crop_indexes, clu_gt_bin_ls, clu_gt_map):
    crop_mat = np.zeros((json_obj.V.shape[0], nn_crop_indexes[0].shape[0]))
    for i in range(nn_crop_indexes[0].shape[0]):
        # 获取当前中心点，获取当前中心点对应的 V 中的点的索引，从 V 中获取对应的点
        point_indices = nn_crop_indexes[0][i] # (1500,)
        label = clu_gt_map[i] # (1,)
        crop_mat[point_indices, i] = label * clu_gt_bin_ls[i, 0, :] # (24000, 1)
    crop_labels = get_max_frequency_non_zero_class(crop_mat.T) # (24000, 1)
    # 使用 gu.print_3d 函数可视化
    gu.print_3d(gu.np_to_pcd_with_label(json_obj.V, crop_labels))

def vis_crop_labels(json_obj, nn_crop_indexes, clu_gt_label):
    crop_gt_mat = np.zeros((json_obj.V.shape[0], nn_crop_indexes[0].shape[0]))
    for i in range(nn_crop_indexes[0].shape[0]):
        # 获取当前中心点，获取当前中心点对应的 V 中的点的索引，从 V 中获取对应的点
        point_indices = nn_crop_indexes[0][i]
        crop_gt_mat[point_indices, i] = clu_gt_label[i, 0, :]
    crop_gt_labels = get_max_frequency_non_zero_class(crop_gt_mat.T)
    # 使用 gu.print_3d 函数可视化
    gu.print_3d(gu.np_to_pcd_with_label(json_obj.V, crop_gt_labels))




if __name__ == '__main__':

    # 1. 读取JSON文件
    # results_path = r'C:\Users\win10\DualMesh\pynb\teeth3ds\model\results\exp_edge_mask_V3_triple_bin_cls\2classes_60epochs_10000ncells_0.005ratio_0.02_rmin0.05_rmax_0.001lr'
    # results_path = r'C:\Users\win10\DualMesh\pynb\teeth3ds\model\results\exp_TripleBinClsNet\2classes_60epochs_10000ncells_0.3ratio_0.03_rmin0.04_rmax_0.001lr'
    results_path = r'C:\Users\win10\DualMesh\pynb\teeth3ds\model\results\exp_TripleBinClsNet\tests'


    json_data_list, pred_accs = read_json_files(results_path)
    # print(f'预测准确率列表: {pred_accs}')

    # 2. 从JSON数据列表中获取特定索引的对象
    id =0
    json_obj = get_json_obj(json_data_list, id)
    # 3. 计算流程
    opts = {
        'alpha': 0.2,
        'max_iter': 5,
        'iter_thred_len': 0.01,
        'field_bound': 0.03,
        'field_sigma': 0.1,
        'num_classes': 15,
        'crop_num': 1280
    }
    outs = topo_cluster(json_obj, opts)
    clu_num, clu_cen, clu_counts, clu_gt_map, counts_mat, gt_clu_num_prop,sep_teeth_mask,clu_labels_all = outs[0:8]

    # 定义新的 namedtuple 类型，添加新的字段
    json_obj_ext = namedtuple('json_obj_ext', json_obj._fields + ('clu_num', 'clu_cen', 'clu_counts', 'clu_gt_map', 'counts_mat', 'gt_clu_num_prop', 'sep_teeth_mask', 'clu_labels_all'))
    # 创建新的 namedtuple 实例，将原 json_obj 的值和新的值合并
    json_obj = json_obj_ext(*json_obj, clu_num=clu_num, clu_cen=clu_cen, clu_counts=clu_counts, clu_gt_map=clu_gt_map, counts_mat=counts_mat, gt_clu_num_prop=gt_clu_num_prop, sep_teeth_mask=sep_teeth_mask, clu_labels_all=clu_labels_all)

    pred_clu = {}
    pred_clu['clu_labels_all'] = torch.from_numpy(clu_labels_all)
    pred_clu['clu_num'] = torch.tensor(clu_num)
    pred_clu['clu_cen'] = torch.from_numpy(clu_cen)
    pred_clu['clu_counts'] = torch.from_numpy(clu_counts)
    pred_clu['clu_gt_map'] = torch.from_numpy(clu_gt_map)
    pred_clu['counts_mat'] = torch.from_numpy(counts_mat)
    pred_clu['gt_clu_num_prop'] = torch.from_numpy(gt_clu_num_prop)
    pred_clu['sep_teeth_mask'] = torch.from_numpy(sep_teeth_mask)

    pred_clu = get_patches(pred_clu, json_obj.V, json_obj.gt_labels, opts)

    nn_crop_indexes = pred_clu['nn_crop_indexes']
    clu_gt_bin_ls = pred_clu['clu_gt_bin_ls']
    clu_gt_label = pred_clu['clu_gt_label']
    overlap_ratio_mat = pred_clu['overlap_ratio_mat']

    # pca_comps, shape_feats, obb_extents = outs[8:]
    np.set_printoptions(precision=4, suppress=True)
    print("gt_clu_num_prop:")
    for row in gt_clu_num_prop:
        formatted_row = " ".join([f"{val:8.4f}" for val in row])
        print(formatted_row)
    # 打印重合率矩阵
    print("nn_crop_indexes 两两之间的重合率矩阵：")
    for row in overlap_ratio_mat:
        formatted_row = " ".join([f"{val:8.4f}" for val in row])
        print(formatted_row)

    print(clu_gt_map)
    print(counts_mat)


    # gu.print_3d(gu.np_to_pcd_with_label(json_obj.V, sep_teeth_mask))
    # gu.print_3d(gu.np_to_pcd_with_label(json_obj.V, clu_labels_all+1))

    # feat = np.concatenate([clu_cen, pca_comps[:,0:2], shape_feats, obb_extents], axis=1)
    # vis_crop_bin_labels(json_obj, nn_crop_indexes, clu_gt_bin_ls, clu_gt_map)
    # vis_crop_labels(json_obj, nn_crop_indexes, clu_gt_label)
    # vis_clu_geom()

    # # 查看每一个crop_indexes
    # for i in range(nn_crop_indexes[0].shape[0]):
    #     # 获取当前中心点，获取当前中心点对应的 V 中的点的索引，从 V 中获取对应的点
    #     center_point = clu_cen[i]
    #     point_indices = nn_crop_indexes[0][i]
    #     points = json_obj.V[point_indices]
    #
    #     # 可以选择将中心点添加到要可视化的点集中
    #     points_with_center = np.vstack([center_point, json_obj.V])
    #     crop_labels = np.zeros(json_obj.V.shape[0])
    #     crop_labels[point_indices] = 1
    #     # 使用 gu.print_3d 函数可视化
    #     combined_labels = np.concatenate([np.array([3]), crop_labels])
    #     gu.print_3d(gu.np_to_pcd_with_label(points_with_center, combined_labels))

    # # cluster_gt_seg_label
    # for i in range(nn_crop_indexes[0].shape[0]):
    #     # 获取当前中心点，获取当前中心点对应的 V 中的点的索引，从 V 中获取对应的点
    #     center_point = cluster_cen[i]
    #     point_indices = nn_crop_indexes[0][i]
    #     points = V[point_indices]
    #
    #     # 可以选择将中心点添加到要可视化的点集中
    #     # points_with_center = np.vstack([center_point, points])
    #     crop_labels = np.zeros(V.shape[0])
    #     crop_labels[point_indices] = cluster_gt_seg_label[i,0,:]
    #     # 使用 gu.print_3d 函数可视化
    #     gu.print_3d(gu.np_to_pcd_with_label(V, crop_labels))

