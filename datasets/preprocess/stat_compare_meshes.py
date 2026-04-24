import os
import numpy as np
from pathlib import Path
import csv
import struct
import pymeshlab
import external_libs.gen_utils as gu
import open3d as o3d

def read_mesh_vertices(file_path):

    file_path = Path(file_path)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file_path))
    mesh = ms.current_mesh()
    mesh.compact()

    vertices = mesh.vertex_matrix()

    return vertices

def calculate_mesh_statistics(vertices, y_axis_min= -36.9843781139949, y_axis_max= 33.15232091532151):
    """
    计算网格的统计信息

    Args:
        vertices (numpy.ndarray): 顶点坐标数组 (N, 3)

    Returns:
        dict: 包含各种统计信息的字典
    """
    if len(vertices) == 0:
        return None

    y_axis_range = y_axis_max - y_axis_min

    # 基本统计
    vertex_count = len(vertices)

    # 坐标范围
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    z_coords = vertices[:, 2]

    # 统计信息
    stats = {
        'vertex_count': vertex_count,
        'x_mean': np.mean(x_coords),
        'y_mean': np.mean(y_coords),
        'z_mean': np.mean(z_coords),
        'x_std': np.std(x_coords),
        'y_std': np.std(y_coords),
        'z_std': np.std(z_coords),
        'xmin': np.min(x_coords),
        'xmax': np.max(x_coords),
        'ymin': np.min(y_coords),
        'ymax': np.max(y_coords),
        'zmin': np.min(z_coords),
        'zmax': np.max(z_coords),
        'cxmin': (np.min(x_coords) - np.mean(x_coords)),  # 减去中心点后的坐标范围
        'cxmax': (np.max(x_coords) - np.mean(x_coords)),
        'cymin': (np.min(y_coords) - np.mean(y_coords)),
        'cymax': (np.max(y_coords) - np.mean(y_coords)),
        'czmin': (np.min(z_coords) - np.mean(z_coords)),
        'czmax': (np.max(z_coords) - np.mean(z_coords))
        # 'cxmin': (np.min(x_coords)-np.mean(x_coords)-y_axis_min)/y_axis_range,   # 减去中心点后的坐标范围
        # 'cxmax': (np.max(x_coords)-np.mean(x_coords)-y_axis_min)/y_axis_range,
        # 'cymin': (np.min(y_coords)-np.mean(y_coords)-y_axis_min)/y_axis_range,
        # 'cymax': (np.max(y_coords)-np.mean(y_coords)-y_axis_min)/y_axis_range,
        # 'czmin': (np.min(z_coords)-np.mean(z_coords)-y_axis_min)/y_axis_range,
        # 'czmax': (np.max(z_coords)-np.mean(z_coords)-y_axis_min)/y_axis_range
    }

    return stats


def analyze_mesh_files_in_directory(root_path, output_base_path):
    """
    统计目录下所有网格文件(OBJ/STL)的信息并保存到两个CSV文件

    Args:
        root_path (str): 根目录路径
        output_base_path (str): 输出文件基础路径（不含扩展名）
    """
    root_path = Path(root_path)

    # 查找所有OBJ和STL文件
    obj_files = list(root_path.rglob("*.obj"))
    stl_files = list(root_path.rglob("*.stl"))
    all_files = obj_files + stl_files

    print(f"找到 {len(obj_files)} 个OBJ文件")
    print(f"找到 {len(stl_files)} 个STL文件")
    print(f"总共 {len(all_files)} 个网格文件")

    # 存储所有统计信息
    all_stats = []

    # 统计总体信息
    all_x_means = []
    all_y_means = []
    all_z_means = []
    all_x_stds = []
    all_y_stds = []
    all_z_stds = []

    # 处理每个网格文件
    for i, mesh_file in enumerate(all_files, 1):
        try:
            file_type = mesh_file.suffix.upper()
            print(f"处理文件 {i}/{len(all_files)}: {mesh_file.name} ({file_type})")

            # 读取顶点
            vertices = read_mesh_vertices(mesh_file)

            if vertices.size == 0:
                print(f"  跳过空文件: {mesh_file.name}")
                continue

            # 计算统计信息
            stats = calculate_mesh_statistics(vertices)

            if stats:
                # 添加文件信息
                stats['filename'] = mesh_file.name
                stats['file_type'] = file_type[1:]  # 去掉点号
                stats['relative_path'] = str(mesh_file.relative_to(root_path))
                all_stats.append(stats)

                # 收集总体统计
                all_x_means.append(stats['x_mean'])
                all_y_means.append(stats['y_mean'])
                all_z_means.append(stats['z_mean'])
                all_x_stds.append(stats['x_std'])
                all_y_stds.append(stats['y_std'])
                all_z_stds.append(stats['z_std'])

                print(f"  顶点数: {stats['vertex_count']}")
                print(f"  均值: ({stats['x_mean']:.3f}, {stats['y_mean']:.3f}, {stats['z_mean']:.3f})")
                print(f"  标准差: ({stats['x_std']:.3f}, {stats['y_std']:.3f}, {stats['z_std']:.3f})")
                print(f"  范围: X[{stats['xmin']:.3f}, {stats['xmax']:.3f}] "
                      f"Y[{stats['ymin']:.3f}, {stats['ymax']:.3f}] "
                      f"Z[{stats['zmin']:.3f}, {stats['zmax']:.3f}]")
                print(f"  去中心后的范围: X[{stats['cxmin']:.3f}, {stats['cxmax']:.3f}] "
                      f"Y[{stats['cymin']:.3f}, {stats['cymax']:.3f}] "
                      f"Z[{stats['czmin']:.3f}, {stats['czmax']:.3f}]")

        except Exception as e:
            print(f"  错误处理文件 {mesh_file.name}: {e}")

    # 计算总体统计信息
    vertex_counts = [s['vertex_count'] for s in all_stats]
    total_vertices = sum(vertex_counts)

    # 加权平均计算
    weighted_means = {}
    for coord in ['x', 'y', 'z']:
        coord_means = [s[f'{coord}_mean'] for s in all_stats]
        coord_stds = [s[f'{coord}_std'] for s in all_stats]

        # 加权平均值
        weighted_means[f'{coord}_mean'] = (
            sum(mean * count for mean, count in zip(coord_means, vertex_counts)) / total_vertices
            if total_vertices > 0 else 0
        )

        # 加权标准差（简化处理）
        weighted_means[f'{coord}_std'] = (
            sum(std * count for std, count in zip(coord_stds, vertex_counts)) / total_vertices
            if total_vertices > 0 else 0
        )

    overall_stats = {
        'filename': 'TOTAL_AVERAGE',
        'file_type': '',
        'relative_path': '',
        'vertex_count': total_vertices,
        'x_mean': weighted_means['x_mean'],
        'y_mean': weighted_means['y_mean'],
        'z_mean': weighted_means['z_mean'],
        'x_std': weighted_means['x_std'],
        'y_std': weighted_means['y_std'],
        'z_std': weighted_means['z_std'],
        'xmin': np.min([s['xmin'] for s in all_stats]) if all_stats else 0,
        'xmax': np.max([s['xmax'] for s in all_stats]) if all_stats else 0,
        'ymin': np.min([s['ymin'] for s in all_stats]) if all_stats else 0,
        'ymax': np.max([s['ymax'] for s in all_stats]) if all_stats else 0,
        'zmin': np.min([s['zmin'] for s in all_stats]) if all_stats else 0,
        'zmax': np.max([s['zmax'] for s in all_stats]) if all_stats else 0,
        'cxmin': np.min([s['cxmin'] for s in all_stats]) if all_stats else 0,
        'cxmax': np.max([s['cxmax'] for s in all_stats]) if all_stats else 0,
        'cymin': np.min([s['cymin'] for s in all_stats]) if all_stats else 0,
        'cymax': np.max([s['cymax'] for s in all_stats]) if all_stats else 0,
        'czmin': np.min([s['czmin'] for s in all_stats]) if all_stats else 0,
        'czmax': np.max([s['czmax'] for s in all_stats]) if all_stats else 0
    }

    # 定义CSV文件路径
    single_csv_path = f"{output_base_path}_single.csv"
    all_csv_path = f"{output_base_path}_all.csv"

    # CSV字段定义
    fieldnames = [
        'filename', 'file_type', 'relative_path', 'vertex_count',
        'x_mean', 'y_mean', 'z_mean',
        'x_std', 'y_std', 'z_std',
        'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax',
        'cxmin', 'cxmax', 'cymin', 'cymax', 'czmin', 'czmax'
    ]

    # 保存单个文件统计信息
    with open(single_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 写入每个文件的统计信息
        for stats in all_stats:
            writer.writerow(stats)

    # 保存总体统计信息
    with open(all_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 写入总体平均统计
        writer.writerow(overall_stats)

    print(f"\n🎉 统计完成!")
    print(f"处理了 {len(all_stats)} 个网格文件")
    print(f"其中 OBJ文件: {len([s for s in all_stats if s['file_type'] == 'obj'])} 个")
    print(f"其中 STL文件: {len([s for s in all_stats if s['file_type'] == 'stl'])} 个")
    print(f"结果已保存到:")
    print(f"  单个文件统计: {single_csv_path}")
    print(f"  总体统计: {all_csv_path}")
    print(f"\n总体统计:")
    print(f"  平均X均值: {overall_stats['x_mean']:.3f}")
    print(f"  平均Y均值: {overall_stats['y_mean']:.3f}")
    print(f"  平均Z均值: {overall_stats['z_mean']:.3f}")
    print(f"  总体范围: X[{overall_stats['xmin']:.3f}, {overall_stats['xmax']:.3f}] "
          f"Y[{overall_stats['ymin']:.3f}, {overall_stats['ymax']:.3f}] "
          f"Z[{overall_stats['zmin']:.3f}, {overall_stats['zmax']:.3f}]")
    print(f"  去中心后的总体范围: X[{overall_stats['cxmin']:.3f}, {overall_stats['cxmax']:.3f}] "
          f"Y[{overall_stats['cymin']:.3f}, {overall_stats['cymax']:.3f}] "
          f"Z[{overall_stats['czmin']:.3f}, {overall_stats['czmax']:.3f}]")

def test_stat():
    global root
    # root = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori_mini"
    # root = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori"
    # root = r"E:\mesh_datasets\3DTeeth\data"
    root = r"E:\mesh_datasets\3DTeethChallenge\data_obj_parent_directory"
    output_csv_path = root + r"\t3ds_stat_yaxis"
    # 执行统计分析
    print("=" * 60)
    print("开始统计网格文件信息(OBJ/STL)...")
    print("=" * 60)
    analyze_mesh_files_in_directory(root, output_csv_path)

def preprocess_method1():
    """
    结论
    1. 两个数据集的牙齿尺度是一致的，使用统一的去中心化处理，可以对齐各种牙齿；
    2. 3DTeeth数据集无论lower及upper的Z轴是朝上的。
    3. TADPM数据集的lower数Z轴是朝下的，而upper数Z轴是朝上的。
    """
    # mesh_path1 = r"E:\mesh_datasets\3DTeeth\data\lower\0LF355FQ\0LF355FQ_lower.obj"
    # mesh_path2 = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori\0002\ori\0002_L_Ori.stl"

    patient_id1 = "0LF355FQ"
    is_lower1 = True
    root1 = r"E:\mesh_datasets\3DTeeth\data"
    def concat_path_fn1(root, patient_id, is_lower):
        return os.path.join(root1, "lower" if is_lower else "upper", patient_id, f"{patient_id}_{'lower' if is_lower else 'upper'}.obj")
    mesh_path1 = concat_path_fn1(root1, patient_id1, is_lower1)

    patient_id2 = "0005"
    is_lower2 = True
    root2= r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori"
    def concat_path_fn2(root, patient_id, is_lower):
        return os.path.join(root, patient_id, "ori", f"{patient_id}_{'L' if is_lower else 'U'}_Ori.stl")
    mesh_path2 = concat_path_fn2(root2, patient_id2, is_lower2)


    points1 = read_mesh_vertices(mesh_path1)
    points2 = read_mesh_vertices(mesh_path2)

    def normalize_fn1(points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        # max_length = np.max(np.linalg.norm(points, axis=1))
        points /= 60.0
        return points

    def normalize_fn2(points,is_lower):
        centroid = np.mean(points, axis=0)
        points -= centroid
        # zmax = np.max(points[:, 2])
        # points[:, 2] = centroid[2] - points[:, 2] if is_lower else points[:, 2]
        points /= 60.0
        return points

    points1 = normalize_fn1(points1)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    points2 = normalize_fn2(points2, is_lower2)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.translate([0,0,0.2])

    gu.print_3d(pcd1,pcd2)



if __name__ == '__main__':

    # test_stat()
    preprocess_method1()