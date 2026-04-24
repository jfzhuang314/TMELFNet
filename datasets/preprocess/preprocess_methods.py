
import os
from pathlib import Path
import pymeshlab
from sklearn.decomposition import PCA

import external_libs.gen_utils as gu
import open3d as o3d
import numpy as np

def read_mesh_vertices(file_path):

    file_path = Path(file_path)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file_path))
    mesh = ms.current_mesh()
    mesh.compact()

    vertices = mesh.vertex_matrix()

    return vertices


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

def preprocess_method2():
    """
    结论
    1. 两个数据集的牙齿尺度是一致的，使用统一的去中心化处理，可以对齐各种牙齿；
    2. 3DTeeth数据集无论lower及upper的Z轴是朝上的。
    3. TADPM数据集的lower数Z轴是朝下的，而upper数Z轴是朝上的。
    """
    # mesh_path1 = r"E:\mesh_datasets\3DTeeth\data\lower\0LF355FQ\0LF355FQ_lower.obj"
    # mesh_path2 = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori\0002\ori\0002_L_Ori.stl"
    # 1MWJLE4X_upper
    # patient_id, is_lower = "1MWJLE4X", False
    # JJEIMAPR_lower
    patient_id, is_lower = "JJEIMAPR", True
    # JKLNUYPM_lower
    # patient_id, is_lower = "JKLNUYPM", True
    # K70NYZYE_upper
    # patient_id, is_lower = "K70NYZYE", False
    # LUQGC700_upper
    # patient_id, is_lower = "LUQGC700", False
    # MGYSCSI9_lower
    # patient_id, is_lower = "MGYSCSI9", True
    # 6L4N5MA2_lower
    # patient_id, is_lower = "6L4N5MA2", True

    root1 = r"E:\mesh_datasets\3DTeeth\data"
    def concat_path_fn1(root, patient_id, is_lower):
        return os.path.join(root1, "lower" if is_lower else "upper", patient_id, f"{patient_id}_{'lower' if is_lower else 'upper'}.obj")
    mesh_path1 = concat_path_fn1(root1, patient_id, is_lower)


    points1 = read_mesh_vertices(mesh_path1)

    def normalize_fn1(points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        # max_length = np.max(np.linalg.norm(points, axis=1))
        points /= 60.0
        return points


    points1 = normalize_fn1(points1)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    def visualize_pca_open3d(points1):
        """
        使用Open3D进行PCA可视化
        """
        # 转换为numpy
        if hasattr(points1, 'detach'):
            points_np = points1.detach().cpu().numpy()
        else:
            points_np = np.array(points1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np[::10])  # 稀疏采样

        # PCA分析
        # 1. 学习XOY平面的主成分
        points_2d = points_np[:, :2]  # 提取XOY平面坐标
        pca_2d = PCA(n_components=2)
        pca_2d.fit(points_2d)

        # 2. 构造3D主成分矩阵
        # 前两行来自XOY平面PCA，第三行为Z轴
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:2, :2] = pca_2d.components_  # XOY平面主成分
        rotation_matrix[2, 2] = 1.0  # Z轴分量

        Rz_180 = np.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                ])


        # rotation_matrix = Rz_180 @ rotation_matrix

        print("3D旋转矩阵:")
        print(rotation_matrix)
        print(f"矩阵行列式: {np.linalg.det(rotation_matrix):.6f}")

        # 3. 验证正交性
        print("正交性验证:")
        print(f"R @ R.T = \n{rotation_matrix @ rotation_matrix.T}")

        # 4. 应用变换到原始点云
        # 将点云中心移到原点
        centroid = np.mean(points_np, axis=0)
        points_centered = points_np - centroid

        # 应用旋转变换
        points_transformed = points_centered @ rotation_matrix.T
        fit_result = fit_parabola_and_analyze(points_transformed)

        if ~is_curve_opening_up(points_transformed):
            rotation_matrix = Rz_180 @ rotation_matrix
            points_transformed = points_centered @ rotation_matrix.T

        pcd_transformed = o3d.geometry.PointCloud()
        pcd_transformed.points = o3d.utility.Vector3dVector(points_transformed[::10])  # 稀疏采样
        pcd_transformed.translate([0,0,0.5])

        # fit_result = fit_parabola_and_analyze(points_transformed)


        # 创建坐标轴
        origin = np.mean(points_np, axis=0)
        scale = np.std(points_np) * 0.5

        # 创建PCA坐标轴线
        axes = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB

        principal_components = rotation_matrix[:, :3]
        for i, pc in enumerate(principal_components):
            axis_line = o3d.geometry.LineSet()
            axis_line.points = o3d.utility.Vector3dVector([
                origin, origin + pc * scale
            ])
            axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            axis_line.colors = o3d.utility.Vector3dVector([colors[i]])
            axes.append(axis_line)

        # 创建世界坐标轴
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale*1.2, origin=origin
        )

        # 可视化
        geometries = [pcd, pcd_transformed, world_axes] + axes
        o3d.visualization.draw_geometries(geometries,
                                        window_name="PCA Analysis",
                                        width=1000, height=800)


    def is_curve_opening_up(points_transformed):
        """
        判断XOY平面内曲线是否开口向上
        Args:
            points_transformed: 变换后的3D点云 (N, 3)
        Returns:
            bool: True表示开口向上，False表示开口向下
        """
        # 提取XOY平面坐标
        x = points_transformed[:, 0]
        y = points_transformed[:, 1]

        # 找到数据的主要延伸方向
        cov_matrix = np.cov(x, y)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

        # 选择主要延伸方向
        main_direction = eigenvecs[:, np.argmax(eigenvals)]

        # 将点投影到主轴方向作为自变量
        t = x * main_direction[0] + y * main_direction[1]

        # 垂直方向作为因变量
        perpendicular_direction = np.array([-main_direction[1], main_direction[0]])
        z = x * perpendicular_direction[0] + y * perpendicular_direction[1]

        # 拟合二次关系: z = a*t² + b*t + c
        A = np.column_stack([
            t**2,
            t,
            np.ones_like(t)
        ])

        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        a = coeffs[0]  # 二次项系数

        # 返回布尔值：True表示开口向上(a>0)，False表示开口向下(a<0)
        return a > 0

    def fit_parabola_and_analyze(points_transformed):
        """
        分析XOY平面内二次关系的完整版本
        """
        is_up = is_curve_opening_up(points_transformed)

        print(f"曲线开口方向: {'向上' if is_up else '向下'}")

        return {
            'is_opening_up': is_up,
            'description': '开口向上' if is_up else '开口向下'
        }

    # 使用示例
    visualize_pca_open3d(points1)

    # gu.print_3d(pcd1)

# def preprocess_method3():



if __name__ == '__main__':

    # test_stat()
    preprocess_method2()