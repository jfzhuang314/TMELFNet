import os
from copy import deepcopy

import numpy as np
import open3d as o3d
from external_libs.mesh_utils import *
import external_libs.pv_gen_utils as pvgu
import pyvista as pv
from PIL import Image as PILImage


def crop_image_array_whitespace(img_array, margin=10, bg_color=(255, 255, 255)):
    """
    裁剪 numpy 数组图片中的空白区域

    Args:
        img_array: numpy 数组 (H, W, 3) 或 (H, W, 4)
        margin: 裁剪后保留的边缘空白像素（默认 10 像素）
        bg_color: 背景颜色 RGB 值（默认白色）

    Returns:
        裁剪后的 numpy 数组
    """
    # 确保是 numpy 数组
    img_array = np.array(img_array)

    # 处理 RGBA 转 RGB
    if img_array.shape[2] == 4:
        img_rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:] / 255.0
        bg_array = np.array(bg_color, dtype=np.uint8)
        img_array = (img_rgb * alpha + bg_array * (1 - alpha)).astype(np.uint8)

    # 找到非背景像素的边界
    non_bg_mask = ~np.all(img_array == bg_color, axis=2)

    # 如果整个图片都是背景，直接返回
    if not np.any(non_bg_mask):
        print("⚠️ 图片全是背景色，无需裁剪")
        return img_array

    # 获取非背景区域的边界框
    coords = np.argwhere(non_bg_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 添加边缘空白
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(img_array.shape[0], y_max + margin + 1)
    x_max = min(img_array.shape[1], x_max + margin + 1)

    # 裁剪图片
    cropped_img = img_array[y_min:y_max, x_min:x_max]

    return cropped_img

def crop_png_whitespace(image_path, output_path=None, margin=10, bg_color=(255, 255, 255)):
    """
    裁剪 PNG 图片中的空白区域

    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径（如果为 None，则覆盖原文件）
        margin: 裁剪后保留的边缘空白像素（默认 10 像素）
        bg_color: 背景颜色 RGB 值（默认白色）

    Returns:
        裁剪后的图片路径
    """
    # 打开图片
    img = PILImage.open(image_path)

    # 转换为 RGB 模式（如果是 RGBA 则去掉透明度）
    if img.mode == 'RGBA':
        # 创建白色背景
        background = PILImage.new('RGB', img.size, bg_color)
        background.paste(img, mask=img.split()[3])  # 使用透明度通道作为 mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # 转换为 numpy 数组
    img_array = np.array(img)

    # 找到非背景像素的边界
    # 检查每个像素是否与背景色相同
    if len(bg_color) == 3:
        non_bg_mask = ~np.all(img_array == bg_color, axis=2)
    else:
        non_bg_mask = ~np.all(img_array == bg_color[:3], axis=2)

    # 如果整个图片都是背景，直接返回
    if not np.any(non_bg_mask):
        print("⚠️ 图片全是背景色，无需裁剪")
        return image_path

    # 获取非背景区域的边界框
    coords = np.argwhere(non_bg_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 添加边缘空白
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(img_array.shape[0], y_max + margin + 1)
    x_max = min(img_array.shape[1], x_max + margin + 1)

    # 裁剪图片
    cropped_img = img.crop((x_min, y_min, x_max, y_max))

    # 保存裁剪后的图片
    if output_path is None:
        output_path = image_path

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cropped_img.save(output_path)

    # 打印裁剪信息
    original_size = img.size
    cropped_size = cropped_img.size
    reduction_x = (1 - cropped_size[0] / original_size[0]) * 100
    reduction_y = (1 - cropped_size[1] / original_size[1]) * 100

    print(f"✅ 图片裁剪完成")
    print(f"   原始尺寸：{original_size[0]} x {original_size[1]}")
    print(f"   裁剪后尺寸：{cropped_size[0]} x {cropped_size[1]}")
    print(f"   宽度减少：{reduction_x:.1f}%")
    print(f"   高度减少：{reduction_y:.1f}%")
    print(f"   保存路径：{output_path}")

    return cropped_img



class MeshFactory:
    @staticmethod
    def create_mesh_from_pygdata(data_obj, color=[0.5, 0.5, 0.5]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(data_obj.pos.cpu().numpy().astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(data_obj.face.T.cpu().numpy().astype(np.int32))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh

    @staticmethod
    def create_raw_mesh(V, F,color=[0.5, 0.5, 0.5]):

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(F)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh

    @staticmethod
    def create_edge_mesh(V, F, edge_mask, color=[1.0, 0.0, 0.0]):
        edge_faces = get_filtered_faces(edge_mask.cpu().numpy(), F)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(edge_faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    @staticmethod
    def create_tooth_mesh(V, F, bg_mask, color=[0.0, 0.0, 1.0]):

        tooth_faces = get_filtered_faces(bg_mask.cpu().numpy(), F)  # tooth faces

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(tooth_faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    @staticmethod
    def create_sep_teeth_mesh(V, F, bg_mask, edge_mask, color=[1.0, 1.0, 0.0]):

        tooth_faces = get_filtered_faces(bg_mask.cpu().numpy(), F)  # tooth faces
        tae_faces = get_filtered_faces(edge_mask.cpu().numpy(), F)  # tae faces
        sep_teeth_faces = setdiff(tooth_faces, tae_faces)  # sep-teeth faces

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(sep_teeth_faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh





class TMCVis:

    @staticmethod
    def vis_mesh_grid(mesh_list, rows=None, cols=None, x_offset=1.2, y_offset=1.2,
                      title_prefix="Mesh",
                      save_path="vis_mesh_grid.png",
                      mesh_show_wireframe = False,
                      mesh_show_back_face = False):
        """
        将多个 mesh 按网格布局排列并可视化，自动计算每个 mesh 的 x,y 偏移

        Args:
            mesh_list: mesh 列表，每个元素可以是 TriangleMesh 或 (mesh, title) 元组
            rows: 行数，如果为 None 则自动计算
            cols: 列数，如果为 None 则自动计算
            x_offset: x 方向偏移量
            y_offset: y 方向偏移量
            title_prefix: 标题前缀

        Returns:
            无返回值，直接调用 gu.print_3d 进行可视化

        Example:
            # 简单用法
            vis_mesh_grid([mesh1, mesh2, mesh3, mesh4])

            # 自定义行列数和标题
            meshes = [(mesh1, "原始"), (mesh2, "边缘"), (mesh3, "牙齿"), (mesh4, "分离")]
            vis_mesh_grid(meshes, rows=2, cols=2, title_prefix="结果")
        """
        n_meshes = len(mesh_list)

        if n_meshes == 0:
            print("⚠️ mesh_list 为空")
            return

        # 自动计算行列数
        if rows is None and cols is None:
            cols = int(np.ceil(np.sqrt(n_meshes)))
            rows = int(np.ceil(n_meshes / cols))
        elif rows is None:
            rows = int(np.ceil(n_meshes / cols))
        elif cols is None:
            cols = int(np.ceil(n_meshes / rows))

        print(f"📊 网格布局：{rows}行 × {cols}列，共 {n_meshes} 个 mesh")

        # 存储所有变换后的 mesh
        transformed_meshes = []

        for idx, mesh_item in enumerate(mesh_list):
            # 处理输入：支持 mesh 或 (mesh, title) 元组
            if isinstance(mesh_item, tuple):
                mesh, custom_title = mesh_item
            else:
                mesh = mesh_item
                custom_title = None

            if mesh is None:
                print(f"⚠️ mesh_list 中第 {idx + 1} 个元素为 None")
                continue

            # 计算当前 mesh 在网格中的位置
            row = idx // cols
            col = idx % cols

            # 计算平移量（以网格中心为原点）
            center_x = (col - (cols - 1) / 2) * x_offset
            center_y = (row - (rows - 1) / 2) * (-y_offset)  # y 轴向下为正

            # 创建平移矩阵
            translation = np.array([[1, 0, 0, center_x],
                                    [0, 1, 0, center_y],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

            # 复制并变换 mesh
            mesh_transformed = mesh.clone() if hasattr(mesh, 'clone') else deepcopy(mesh)
            mesh_transformed.transform(translation)

            # 生成标题
            if custom_title:
                title = f"{title_prefix}_{custom_title}" if title_prefix else custom_title
            else:
                title = f"{title_prefix}_{idx + 1}" if title_prefix else f"Mesh_{idx + 1}"

            transformed_meshes.append((mesh_transformed, title))

            print(f"  [{idx + 1}/{n_meshes}] {title}: 位置 ({row},{col}), 平移 ({center_x:.3f}, {center_y:.3f})")

        # 提取所有 mesh 和标题
        meshes_to_vis = [m[0] for m in transformed_meshes]
        titles = [m[1] for m in transformed_meshes]

        # 调用 gu.print_3d 进行可视化
        print(f"\n🎨 调用 gu.print_3d 可视化 {len(meshes_to_vis)} 个 mesh...")
        # gu.print_3d(*meshes_to_vis)
        # gu.print_3d_args(*meshes_to_vis, mesh_show_wireframe=mesh_show_wireframe,
        #              mesh_show_back_face=mesh_show_back_face, save_path=None)
        pvgu.print_3d_args_hdpi(*meshes_to_vis, mesh_show_wireframe=mesh_show_wireframe,
                             mesh_show_back_face=mesh_show_back_face, save_path=save_path,
                              resolution=(int(3840*2), int(2160*2)))
        print("✅ 可视化完成")

        return transformed_meshes


    @staticmethod
    def plot_save_meshes(V, F, bg_mask, edge_mask, tae_mask, tte_mask,
                         verts_clu_labels,
                         gt_labels = None,
                         pred_labels = None,
                         knn_ica_meshes=None,
                         save_path="vis_mesh_grid.png"):
        gt_mesh = MeshFactory.create_raw_mesh(V, F, color=[0.5, 0.5, 0.5])
        pred_mesh = MeshFactory.create_raw_mesh(V, F, color=[0.5, 0.5, 0.5])
        raw_mesh = MeshFactory.create_raw_mesh(V, F, color=[0.5, 0.5, 0.5])
        edge_mesh = MeshFactory.create_edge_mesh(V, F, edge_mask, color=[1.0, 0.0, 0.0])
        tooth_mesh = MeshFactory.create_tooth_mesh(V, F, bg_mask, color=[0.0, 0.0, 1.0])
        tae_sep_teeth_mesh = MeshFactory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask, color=[1.0, 0.0, 0.0])
        tae_tte_sep_teeth_mesh = MeshFactory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask + tte_mask,
                                                                   color=[0.0, 1.0, 0.0])
        edge_sep_teeth_mesh = MeshFactory.create_sep_teeth_mesh(V, F, bg_mask, edge_mask, color=[0.0, 0.0, 1.0])
        if gt_labels is not None:
            gt_mesh = pvgu.get_colored_mesh(gt_mesh, gt_labels.astype(np.int32), 'label15')
        if pred_labels is not None:
            pred_mesh = pvgu.get_colored_mesh(pred_mesh, pred_labels.astype(np.int32), 'label15')
        raw_mesh = pvgu.get_colored_mesh(raw_mesh, verts_clu_labels.astype(np.int32), 'label15')
        tooth_mesh = pvgu.get_colored_mesh(tooth_mesh, verts_clu_labels.astype(np.int32), 'label15')
        tae_sep_teeth_mesh = pvgu.get_colored_mesh(tae_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')
        tae_tte_sep_teeth_mesh = pvgu.get_colored_mesh(tae_tte_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')
        edge_sep_teeth_mesh = pvgu.get_colored_mesh(edge_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')


        # gu.print_3d(mesh, tooth_mesh, edge_mesh)
        mesh_list = [(raw_mesh, 'mesh'),
                     (gt_mesh, 'gt_mesh'),
                     (pred_mesh, 'pred_mesh'),
                     (edge_mesh, 'edge_mesh'),
                     (tooth_mesh, 'tooth_mesh'),
                     # (tae_sep_teeth_mesh, 'tae_sep_teeth_mesh'),
                     # (tae_tte_sep_teeth_mesh, 'tae_tte_sep_teeth_mesh'),
                     (edge_sep_teeth_mesh, 'edge_sep_teeth_mesh'),
                     (knn_ica_meshes[0], 'alpha_meshes'),
                     (knn_ica_meshes[1], 'bnd_contour_3d'),
                     (knn_ica_meshes[2], 'countour_3d'),
                     (knn_ica_meshes[3], 'vector_lines'),
                     ] if knn_ica_meshes else [(raw_mesh, 'mesh'),
                                               (gt_mesh, 'gt_mesh'),
                                               (pred_mesh, 'pred_mesh'),
                                                 (edge_mesh, 'edge_mesh'),
                                                 (tooth_mesh, 'tooth_mesh'),
                                                 # (tae_sep_teeth_mesh, 'tae_sep_teeth_mesh'),
                                                 # (tae_tte_sep_teeth_mesh, 'tae_tte_sep_teeth_mesh'),
                                                 (edge_sep_teeth_mesh, 'edge_sep_teeth_mesh'),]
        V_range = V.max(axis=0) - V.min(axis=0)
        TMCVis.vis_mesh_grid(mesh_list,
                      save_path=save_path,
                      x_offset=V_range[0] * 1.1,
                      y_offset=V_range[1] * 1.1)

class PVMeshFactory:
    """PyVista 版本的网格工厂，与 pv_gen_utils.get_colored_mesh 兼容"""

    @staticmethod
    def _create_pv_mesh(V, F):
        """
        创建 PyVista PolyData 网格

        Args:
            V: 顶点数组 (N, 3)
            F: 面片数组 (M, 3)，需要从 0 开始索引

        Returns:
            pyvista.PolyData 对象
        """

        # PyVista 需要 face 格式：[n_vertices, v1, v2, v3, ...]
        # 每个面片前添加顶点数 3
        if F.shape[1] == 3:
            faces_pv = np.hstack([np.full((F.shape[0], 1), 3), F])
        else:
            faces_pv = F

        return pv.PolyData(V, faces_pv)

    @staticmethod
    def create_mesh_from_pygdata(data_obj, color=None):
        """
        从 PyG 数据对象创建 PyVista 网格

        Args:
            data_obj: PyG Data 对象
            color: 颜色（未使用，保留接口一致性）

        Returns:
            pyvista.PolyData 对象
        """
        V = data_obj.pos.cpu().numpy().astype(np.float64)
        F = data_obj.face.T.cpu().numpy().astype(np.int32)

        return PVMeshFactory._create_pv_mesh(V, F)

    @staticmethod
    def create_raw_mesh(V, F, color=None):
        """
        创建原始 PyVista 网格

        Args:
            V: 顶点数组 (N, 3)
            F: 面片数组 (M, 3)
            color: 颜色（未使用，保留接口一致性）

        Returns:
            pyvista.PolyData 对象
        """
        return PVMeshFactory._create_pv_mesh(V, F)

    @staticmethod
    def create_edge_mesh(V, F, edge_mask, color=None):
        edge_faces = get_filtered_faces(edge_mask, F)
        return PVMeshFactory._create_pv_mesh(V, edge_faces)

    @staticmethod
    def create_tooth_mesh(V, F, bg_mask, color=None):
        """
        创建牙齿 PyVista 网格

        Args:
            V: 顶点数组 (N, 3)
            F: 面片数组 (M, 3)
            bg_mask: 背景掩码
            color: 颜色（未使用，保留接口一致性）

        Returns:
            pyvista.PolyData 对象
        """
        tooth_faces = get_filtered_faces(bg_mask, F)  # tooth faces
        return PVMeshFactory._create_pv_mesh(V, tooth_faces)

    @staticmethod
    def create_sep_teeth_mesh(V, F, bg_mask, edge_mask, color=None):
        """
        创建分离牙齿 PyVista 网格

        Args:
            V: 顶点数组 (N, 3)
            F: 面片数组 (M, 3)
            bg_mask: 背景掩码
            edge_mask: 边缘掩码
            color: 颜色（未使用，保留接口一致性）

        Returns:
            pyvista.PolyData 对象
        """
        tooth_faces = get_filtered_faces(bg_mask, F)  # tooth faces
        tae_faces = get_filtered_faces(edge_mask, F)  # tae faces
        sep_teeth_faces = setdiff(tooth_faces, tae_faces)  # sep-teeth faces
        return PVMeshFactory._create_pv_mesh(V, sep_teeth_faces)

class TMCVisPV:
    """PyVista 版本的 TMC 可视化工具"""

    # @staticmethod
    # def _o3d_to_pyvista(o3d_mesh):
    #     """将 Open3D TriangleMesh 转换为 PyVista PolyData"""
    #     if isinstance(o3d_mesh, pv.PolyData):
    #         return o3d_mesh  # 已经是 PyVista
    #
    #     if isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
    #         vertices = np.asarray(o3d_mesh.vertices)
    #         triangles = np.asarray(o3d_mesh.triangles)
    #         faces_pv = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
    #         return pv.PolyData(vertices, faces_pv)
    @staticmethod
    def _o3d_to_pyvista(o3d_geometry):
        """
        将 Open3D 几何对象转换为 PyVista 对象
        支持: TriangleMesh, PointCloud, LineSet

        Args:
            o3d_geometry: Open3D 几何对象

        Returns:
            pyvista.PolyData 或 None
        """
        if isinstance(o3d_geometry, pv.PolyData):
            return o3d_geometry  # 已经是 PyVista

        # TriangleMesh 转换
        if isinstance(o3d_geometry, o3d.geometry.TriangleMesh):
            vertices = np.asarray(o3d_geometry.vertices)
            triangles = np.asarray(o3d_geometry.triangles)
            faces_pv = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
            pv_mesh = pv.PolyData(vertices, faces_pv)

            # 如果有颜色，添加颜色数据
            if o3d_geometry.has_vertex_colors():
                colors = np.asarray(o3d_geometry.vertex_colors) * 255
                pv_mesh.point_data['colors'] = colors.astype(np.uint8)

            return pv_mesh

        # PointCloud 转换
        elif isinstance(o3d_geometry, o3d.geometry.PointCloud):
            points = np.asarray(o3d_geometry.points)
            pv_pcd = pv.PolyData(points)

            # 如果有颜色，添加颜色数据
            if o3d_geometry.has_colors():
                colors = np.asarray(o3d_geometry.colors) * 255
                pv_pcd.point_data['colors'] = colors.astype(np.uint8)

            return pv_pcd

        # LineSet 转换
        elif isinstance(o3d_geometry, o3d.geometry.LineSet):
            try:
                points = np.asarray(o3d_geometry.points)
                lines_o3d = np.asarray(o3d_geometry.lines)

                # 检查 lines 是否为空
                if lines_o3d.size == 0:
                    print("⚠️ LineSet 没有线段")
                    return pv.PolyData(points)

                # Open3D LineSet lines 格式可能是：
                # 格式1: [[idx1, idx2], [idx3, idx4], ...] (每行2个索引)
                # 格式2: [[2, idx1, idx2], [2, idx3, idx4], ...] (每行3个值，第一个是顶点数)

                # 检测格式
                if lines_o3d.ndim == 2 and lines_o3d.shape[1] == 2:
                    # 格式1: 每行只有2个索引，需要添加顶点数前缀
                    n_lines = lines_o3d.shape[0]
                    pv_lines_list = []
                    for i in range(n_lines):
                        pv_lines_list.append(2)  # 每条线2个点
                        pv_lines_list.append(int(lines_o3d[i, 0]))
                        pv_lines_list.append(int(lines_o3d[i, 1]))

                elif lines_o3d.ndim == 2 and lines_o3d.shape[1] == 3:
                    # 格式2: 每行已有顶点数前缀
                    n_lines = lines_o3d.shape[0]
                    pv_lines_list = []
                    for i in range(n_lines):
                        n_pts = int(lines_o3d[i, 0])
                        pv_lines_list.append(n_pts)
                        for j in range(1, n_pts + 1):
                            pv_lines_list.append(int(lines_o3d[i, j]))

                else:
                    # 尝试展平处理
                    print(f"⚠️ LineSet lines 未知格式: shape={lines_o3d.shape}")
                    lines_flat = lines_o3d.flatten()
                    pv_lines_list = []
                    i = 0
                    while i < len(lines_flat):
                        n_pts = int(lines_flat[i])
                        if i + 1 + n_pts <= len(lines_flat):
                            pv_lines_list.append(n_pts)
                            for j in range(1, n_pts + 1):
                                pv_lines_list.append(int(lines_flat[i + j]))
                            i += 1 + n_pts
                        else:
                            break

                pv_lines = np.array(pv_lines_list, dtype=np.int64)

                # 创建 PyVista PolyData
                pv_lineset = pv.PolyData(points, lines=pv_lines)

                # 如果有颜色，添加颜色数据
                if o3d_geometry.has_colors():
                    colors = np.asarray(o3d_geometry.colors) * 255
                    if len(colors) == len(points):
                        pv_lineset.point_data['colors'] = colors.astype(np.uint8)

                return pv_lineset

            except Exception as e:
                print(f"⚠️ LineSet 转换失败: {e}")
                import traceback
                traceback.print_exc()
                return None

        else:
            print(f"⚠️ 不支持的 Open3D 类型: {type(o3d_geometry)}")
            return None


    @staticmethod
    def vis_mesh_grid_pv(mesh_list, rows=None, cols=None, x_offset=1.2, y_offset=1.2,
                        title_prefix="Mesh",
                        save_path="vis_mesh_grid.png",
                        show_edges=False,
                        show_points=False,
                        background='white',
                         view='xy',
                         view_vector=None,
                         elevation=None,
                         azimuth=None,
                        window_size=[3840, 2160]):
        """
        使用 PyVista 将多个 mesh 按网格布局排列并可视化

        Args:
            mesh_list: mesh 列表，每个元素是 (pyvista.PolyData, title) 元组
            rows: 行数，如果为 None 则自动计算
            cols: 列数，如果为 None 则自动计算
            x_offset: x 方向偏移量
            y_offset: y 方向偏移量
            title_prefix: 标题前缀
            save_path: 保存路径
            show_edges: 是否显示网格边
            background: 背景颜色 'white' 或 'black'
            window_size: 窗口大小 [width, height]

        Returns:
            pyvista.Plotter 对象
        """
        import pyvista as pv

        n_meshes = len(mesh_list)

        if n_meshes == 0:
            print("⚠️ mesh_list 为空")
            return None

        # 自动计算行列数
        if rows is None and cols is None:
            cols = int(np.ceil(np.sqrt(n_meshes)))
            rows = int(np.ceil(n_meshes / cols))
        elif rows is None:
            rows = int(np.ceil(n_meshes / cols))
        elif cols is None:
            cols = int(np.ceil(n_meshes / rows))

        print(f"📊 PyVista 网格布局：{rows}行 × {cols}列，共 {n_meshes} 个 mesh")

        # 创建 Plotter
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.background_color = background

        valid_meshes = []
        # 渲染每个 mesh
        for idx, mesh_item in enumerate(mesh_list):
            if isinstance(mesh_item, tuple):
                mesh, custom_title = mesh_item
            else:
                mesh = mesh_item
                custom_title = None

            if mesh is None:
                print(f"⚠️ mesh_list 中第 {idx + 1} 个元素为 None")
                continue

            # 计算当前 mesh 在网格中的位置
            row = idx // cols
            col = idx % cols

            # 计算平移量（以网格中心为原点）
            center_x = (col - (cols - 1) / 2) * x_offset
            center_y = (row - (rows - 1) / 2) * (-y_offset)

            # 确保 mesh 是 PyVista PolyData
            try:
                mesh = TMCVisPV._o3d_to_pyvista(mesh)
            except Exception as e:
                print(f"⚠️ 跳过第 {idx + 1} 个 mesh：转换失败 - {e}")
                continue

            # 检查转换结果
            if mesh is None:
                print(f"⚠️ [{idx + 1}/{n_meshes}] 跳过：转换后 mesh 为 None")
                continue

            mesh_copy = mesh.copy()
            mesh_copy.points += np.array([center_x, center_y, 0])

            # 生成标题
            if custom_title:
                title = f"{title_prefix}_{custom_title}" if title_prefix else custom_title
            else:
                title = f"{title_prefix}_{idx + 1}" if title_prefix else f"Mesh_{idx + 1}"

            # 添加 mesh 到场景
            if 'colors' in mesh_copy.point_data:
                plotter.add_mesh(
                    mesh_copy,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    show_vertices=show_points,
                    smooth_shading=True
                )
            elif 'labels' in mesh_copy.point_data:
                plotter.add_mesh(
                    mesh_copy,
                    scalars='labels',
                    cmap='tab20',
                    show_edges=show_edges,
                    show_vertices=show_points,
                    smooth_shading=True
                )
            else:
                plotter.add_mesh(
                    mesh_copy,
                    show_edges=show_edges,
                    show_vertices=show_points,
                    smooth_shading=True
                )

            # print(f"  [{idx + 1}/{n_meshes}] {title}: 位置 ({row},{col}), 平移 ({center_x:.3f}, {center_y:.3f})")

            valid_meshes.append(mesh_copy)
            print(f"  [{idx + 1}/{n_meshes}] {title}: 位置 ({row},{col}), 平移 ({center_x:.3f}, {center_y:.3f})")

        if len(valid_meshes) == 0:
            print("⚠️ 没有有效的 mesh 可以渲染")
            plotter.close()
            return None

        # ========== 设置相机视角（紧凑布局） ==========

        # 获取所有有效 mesh 的边界框
        all_points = np.vstack([m.points for m in valid_meshes])
        bounds = all_points.max(axis=0) - all_points.min(axis=0)
        center = all_points.mean(axis=0)

        # 计算最大范围（用于确定相机距离）
        max_range = max(bounds[0], bounds[1])

        # 关键修改：减小相机距离系数，使物体更大、空白更少
        # 原始系数是 2.0，现在改为 1.2-1.5 使布局更紧凑
        camera_distance = max_range * 1.2

        plotter = TMCVisPV.set_view(plotter, camera_distance, center,azimuth, elevation, view, view_vector)

        # 设置相机视角为 XY 平面俯视
        # plotter.view_xy()


        # 保存图片
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            plotter.screenshot(save_path, window_size=window_size,scale=2)
            print(f"✅ 高清图片已保存：{save_path}")
            print(f"   分辨率：{window_size[0]}x{window_size[1]} 像素")

            # 自动裁剪空白区域
            cropped_path = crop_png_whitespace(save_path, margin=20)
            if cropped_path != save_path:
                print(f"   裁剪后路径：{cropped_path}")

        plotter.close()
        return plotter

    @staticmethod
    def set_view(plotter, camera_distance, center,azimuth, elevation,  view, view_vector):
        # 设置相机位置
        if view_vector is not None:
            # 自定义视角向量（优先级最高）
            camera_position = [
                center + np.array(view_vector) * camera_distance,
                center,
                [0, 1, 0]
            ]
            plotter.camera_position = camera_position
        elif elevation is not None and azimuth is not None:
            # 自定义仰角和方位角
            plotter.view_xz()
            plotter.camera.elevation = elevation
            plotter.camera.azimuth = azimuth
            plotter.camera.zoom(1.0)
        elif view == 'xy':
            # XY 平面俯视
            camera_position = [
                [center[0], center[1], center[2] + camera_distance],
                center,
                [0, 1, 0]
            ]
            plotter.camera_position = camera_position
            plotter.view_xy()
        elif view == 'xz':
            # XZ 平面侧视
            camera_position = [
                [center[0], center[2] + camera_distance, center[1]],
                center,
                [0, 0, 1]
            ]
            plotter.camera_position = camera_position
            plotter.view_xz()
        elif view == 'yz':
            # YZ 平面侧视
            camera_position = [
                [center[2] + camera_distance, center[1], center[0]],
                center,
                [0, 1, 0]
            ]
            plotter.camera_position = camera_position
            plotter.view_yz()
        elif view == 'iso':
            # 等距视角
            plotter.view_isometric()
        else:
            # 默认视角
            camera_position = [
                [center[0], center[1], center[2] + camera_distance],
                center,
                [0, 1, 0]
            ]
            plotter.camera_position = camera_position

        return plotter

    @staticmethod
    def plot_save_meshes(V, F, bg_mask, edge_mask, tae_mask, tte_mask,
                         verts_clu_labels,
                         gt_labels=None,
                         pred_labels=None,
                         knn_ica_meshes=None,
                         save_path="vis_mesh_grid.png",
                         show_edges=False,
                         background='white',
                         window_size=[3840, 2160]):
        """
        使用 PyVista 绘制并保存网格（推荐版本）

        Args:
            V: 顶点数组 (N, 3)
            F: 面片数组 (M, 3)
            bg_mask: 背景掩码
            edge_mask: 边缘掩码
            tae_mask: TAE 掩码
            tte_mask: TTE 掩码
            verts_clu_labels: 顶点聚类标签
            gt_labels: Ground Truth 标签
            pred_labels: 预测标签
            knn_ica_meshes: KNN ICA 网格列表
            save_path: 保存路径
            show_edges: 是否显示网格边
            background: 背景颜色
            window_size: 窗口大小 [width, height]
        """
        factory = PVMeshFactory

        # 创建基础网格（使用 PyVista 工厂）
        gt_mesh = factory.create_raw_mesh(V, F)
        pred_mesh = factory.create_raw_mesh(V, F)
        raw_mesh = factory.create_raw_mesh(V, F)
        edge_mesh = factory.create_edge_mesh(V, F, edge_mask)
        tooth_mesh = factory.create_tooth_mesh(V, F, bg_mask)
        tae_sep_teeth_mesh = factory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask)
        tae_tte_sep_teeth_mesh = factory.create_sep_teeth_mesh(V, F, bg_mask, tae_mask + tte_mask)
        edge_sep_teeth_mesh = factory.create_sep_teeth_mesh(V, F, bg_mask, edge_mask)

        # 上色（使用 pv_gen_utils）
        if gt_labels is not None:
            gt_mesh = pvgu.get_colored_mesh(gt_mesh, gt_labels.astype(np.int32), 'label15')
        if pred_labels is not None:
            pred_mesh = pvgu.get_colored_mesh(pred_mesh, pred_labels.astype(np.int32), 'label15')
        raw_mesh = pvgu.get_colored_mesh(raw_mesh, verts_clu_labels.astype(np.int32), 'label15')
        tooth_mesh = pvgu.get_colored_mesh(tooth_mesh, verts_clu_labels.astype(np.int32), 'label15')
        tae_sep_teeth_mesh = pvgu.get_colored_mesh(tae_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')
        tae_tte_sep_teeth_mesh = pvgu.get_colored_mesh(tae_tte_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')
        edge_sep_teeth_mesh = pvgu.get_colored_mesh(edge_sep_teeth_mesh, verts_clu_labels.astype(np.int32), 'label15')

        # 构建网格列表
        mesh_list = [
            (raw_mesh, 'raw'),
            (gt_mesh, 'gt'),
            (pred_mesh, 'pred'),
            (edge_mesh, 'edge'),
            (tooth_mesh, 'tooth'),
            (edge_sep_teeth_mesh, 'sep_teeth')
        ]

        # if knn_ica_meshes:
        #     mesh_list.extend([
        #         (knn_ica_meshes[0], 'alpha'),
        #         (knn_ica_meshes[1], 'bnd_contour'),
        #         (knn_ica_meshes[2], 'contour'),
        #         (knn_ica_meshes[3], 'vectors'),
        #     ])
        if knn_ica_meshes:
            # 调试输出
            print(f"🔍 knn_ica_meshes 类型：{type(knn_ica_meshes)}")
            for i, mesh in enumerate(knn_ica_meshes):
                if mesh is None:
                    print(f"   ⚠️ knn_ica_meshes[{i}] 是 None")

            # 过滤掉 None 元素
            valid_knn_meshes = [(mesh, name) for mesh, name in [
                (knn_ica_meshes[0], 'alpha'),
                (knn_ica_meshes[1], 'bnd_contour'),
                (knn_ica_meshes[2], 'contour'),
                (knn_ica_meshes[3], 'vectors'),
            ] if mesh is not None]

            mesh_list.extend(valid_knn_meshes)


        # 计算偏移量
        V_range = V.max(axis=0) - V.min(axis=0)
        x_offset = V_range[0] * 1.1
        y_offset = V_range[1] * 1.1

        # 可视化
        TMCVisPV.vis_mesh_grid_pv(
            mesh_list,
            save_path=save_path,
            x_offset=x_offset,
            y_offset=y_offset,
            show_edges=show_edges,
            background=background,
            window_size=window_size
        )


    @staticmethod
    def render_mesh_to_image(mesh, title=None, window_size=[800, 600],
                             background='white', show_edges=False,
                             view='xy', elevation=None, azimuth=None,
                             view_vector=None, anti_aliasing=True,
                             multi_samples=8, line_smoothing=True,
                             point_smoothing=True, polygon_smoothing=True,
                             point_size=5.0, line_width=2.0,
                             lighting=True, specular=0.3, specular_power=15):

        """
        将单个 mesh 渲染为图片

        Args:
            mesh: PyVista PolyData 对象
            title: 标题文本
            window_size: 窗口大小 [width, height]
            background: 背景颜色
            show_edges: 是否显示边
            view: 预设视角 'xy', 'xz', 'yz', 'iso'
            elevation: 仰角
            azimuth: 方位角
            view_vector: 自定义视角向量
            anti_aliasing: 是否启用抗锯齿
            multi_samples: 多重采样数（4, 8, 16），数值越高质量越好但速度越慢
            line_smoothing: 线条平滑
            point_smoothing: 点平滑
            polygon_smoothing: 多边形平滑

        Returns:
            numpy array (H, W, 3) RGB 图像
        """

        # 防御性检查：如果 mesh 为 None，返回空白图片
        if mesh is None:
            print("⚠️ render_mesh_to_image: mesh 为 None，返回空白图片")
            return np.full((window_size[1], window_size[0], 3),
                           255 if background == 'white' else 0, dtype=np.uint8)

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.background_color = background

        # 启用抗锯齿
        if anti_aliasing:
            plotter.render_window.SetMultiSamples(multi_samples)

        # 判断几何类型并选择合适的渲染方式
        n_cells = mesh.n_cells
        n_points = mesh.n_points

        # PointCloud 渲染（没有面片，只有点）
        if n_cells == 0 and n_points > 0:
            # 点云渲染
            if 'colors' in mesh.point_data:
                plotter.add_points(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    point_size=point_size,
                    render_points_as_spheres=point_smoothing
                )
            else:
                plotter.add_points(
                    mesh,
                    color='red',
                    point_size=point_size,
                    render_points_as_spheres=point_smoothing
                )

        # LineSet 渲染（只有线，没有面片）
        elif n_cells > 0 and hasattr(mesh, 'lines') and mesh.lines is not None:
            # 线段渲染
            if 'colors' in mesh.point_data:
                plotter.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    line_width=line_width,
                    render_lines_as_tubes=line_smoothing
                )
            else:
                plotter.add_mesh(
                    mesh,
                    color='blue',
                    line_width=line_width,
                    render_lines_as_tubes=line_smoothing
                )

        # TriangleMesh 渲染（有面片）
        else:
            # 添加 mesh
            if 'colors' in mesh.point_data:
                plotter.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    edge_color='black' if show_edges else None,
                    smooth_shading=True,
                    specular=specular if lighting else 0.0,
                    specular_power=specular_power,
                    lighting=lighting,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )
            elif 'labels' in mesh.point_data:
                plotter.add_mesh(
                    mesh,
                    scalars='labels',
                    cmap='tab20',
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=specular if lighting else 0.0,
                    specular_power=specular_power,
                    lighting=lighting,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )
            else:
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=specular if lighting else 0.0,
                    specular_power=specular_power,
                    lighting=lighting,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )

        # 启用多边形平滑
        if polygon_smoothing:
            plotter.renderer.UseShading = True

        # 添加标题
        if title:
            plotter.add_text(title, position='lower_edge', font_size=14)

        points = mesh.points
        bounds = points.max(axis=0) - points.min(axis=0)
        center = points.mean(axis=0)
        max_range = max(bounds[0], bounds[1])        # 计算最大范围（用于确定相机距离）
        camera_distance = max_range * 1.2

        plotter = TMCVisPV.set_view(plotter, camera_distance, center,  azimuth, elevation,view, view_vector)

        plotter.hide_axes()

        # 渲染并获取图像
        img_array = plotter.screenshot(return_img=True)
        plotter.close()

        # 裁剪空白区域
        bg_color_tuple = (255, 255, 255) if background == 'white' else (0, 0, 0)
        img_array = crop_image_array_whitespace(img_array, margin=10, bg_color=bg_color_tuple)

        return img_array


    # @staticmethod
    # def render_mesh_to_image(mesh, title=None, window_size=[800, 600],
    #                          background='white', show_edges=False,
    #                          view='xy', elevation=None, azimuth=None,
    #                          view_vector=None):
    #     """
    #     将单个 mesh 渲染为图片
    #
    #     Args:
    #         mesh: PyVista PolyData 对象
    #         title: 标题文本
    #         window_size: 窗口大小 [width, height]
    #         background: 背景颜色
    #         show_edges: 是否显示边
    #         view: 预设视角 'xy', 'xz', 'yz', 'iso'
    #         elevation: 仰角
    #         azimuth: 方位角
    #         view_vector: 自定义视角向量
    #
    #     Returns:
    #         numpy array (H, W, 3) RGB 图像
    #     """
    #     import pyvista as pv
    #
    #     plotter = pv.Plotter(off_screen=True, window_size=window_size)
    #     plotter.background_color = background
    #
    #     # 添加 mesh
    #     if 'colors' in mesh.point_data:
    #         plotter.add_mesh(
    #             mesh,
    #             scalars='colors',
    #             rgb=True,
    #             show_edges=show_edges,
    #             smooth_shading=True
    #         )
    #     elif 'labels' in mesh.point_data:
    #         plotter.add_mesh(
    #             mesh,
    #             scalars='labels',
    #             cmap='tab20',
    #             show_edges=show_edges,
    #             smooth_shading=True
    #         )
    #     else:
    #         plotter.add_mesh(
    #             mesh,
    #             show_edges=show_edges,
    #             smooth_shading=True
    #         )
    #
    #     # 添加标题
    #     if title:
    #         plotter.add_text(title, position='lower_edge', font_size=14)
    #
    #     points = mesh.points
    #
    #     bounds = points.max(axis=0) - points.min(axis=0)
    #     center = points.mean(axis=0)
    #
    #     # 计算最大范围（用于确定相机距离）
    #     max_range = max(bounds[0], bounds[1])
    #
    #     # 关键修改：减小相机距离系数，使物体更大、空白更少
    #     # 原始系数是 2.0，现在改为 1.2-1.5 使布局更紧凑
    #     camera_distance = max_range * 1.2
    #
    #     plotter = TMCVisPV.set_view(plotter, camera_distance, center, azimuth, elevation, view, view_vector)
    #
    #     plotter.hide_axes()
    #
    #     # 渲染并获取图像
    #     img_array = plotter.screenshot(return_img=True)
    #     plotter.close()
    #
    #     img_array = crop_image_array_whitespace(img_array)
    #
    #     return img_array


    # @staticmethod
    # def combine_images_to_grid(img_list, rows=None, cols=None,
    #                            save_path='combined_mesh_grid.png',
    #                            margin=10, bg_color=(255, 255, 255)):
    #     """
    #     将多个图片按网格布局组合成一张大图
    #
    #     Args:
    #         img_list: 图片列表，每个元素是 numpy array (H, W, 3) 或 (H, W, 4)
    #         rows: 行数，None 则自动计算
    #         cols: 列数，None 则自动计算
    #         save_path: 保存路径
    #         margin: 图片之间的边距（像素）
    #         bg_color: 背景颜色 (R, G, B)
    #
    #     Returns:
    #         组合后的图片路径
    #     """
    #     n_imgs = len(img_list)
    #     if n_imgs == 0:
    #         print("⚠️ 图片列表为空")
    #         return None
    #
    #     # 自动计算行列
    #     if rows is None and cols is None:
    #         cols = int(np.ceil(np.sqrt(n_imgs)))
    #         rows = int(np.ceil(n_imgs / cols))
    #     elif rows is None:
    #         rows = int(np.ceil(n_imgs / cols))
    #     elif cols is None:
    #         cols = int(np.ceil(n_imgs / rows))
    #
    #     print(f"📊 图片网格布局：{rows}行 × {cols}列，共 {n_imgs} 张图片")
    #
    #     # 统一图片尺寸（使用最大尺寸）
    #     max_h = max(img.shape[0] for img in img_list)
    #     max_w = max(img.shape[1] for img in img_list)
    #
    #     # 创建空白画布
    #     total_h = rows * max_h + (rows + 1) * margin
    #     total_w = cols * max_w + (cols + 1) * margin
    #
    #     combined_img = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    #
    #     # 粘贴每张图片
    #     for idx, img in enumerate(img_list):
    #         row = idx // cols
    #         col = idx % cols
    #
    #         # 计算位置
    #         y_pos = margin + row * (max_h + margin)
    #         x_pos = margin + col * (max_w + margin)
    #
    #         # 调整图片尺寸
    #         if img.shape[0] != max_h or img.shape[1] != max_w:
    #             img_pil = PILImage.fromarray(img)
    #             img_resized = img_pil.resize((max_w, max_h), PILImage.LANCZOS)
    #             img = np.array(img_resized)
    #
    #         # 处理 alpha 通道
    #         if img.shape[2] == 4:
    #             img_rgb = img[:, :, :3]
    #             alpha = img[:, :, 3:] / 255.0
    #             img_final = (img_rgb * alpha + bg_color * (1 - alpha)).astype(np.uint8)
    #         else:
    #             img_final = img
    #
    #         # 粘贴到画布
    #         combined_img[y_pos:y_pos+max_h, x_pos:x_pos+max_w] = img_final
    #         print(f"  [{idx + 1}/{n_imgs}] 已放置到位置 ({row}, {col})")
    #
    #     # 保存图片
    #     if save_path:
    #         save_dir = os.path.dirname(save_path)
    #         if save_dir and not os.path.exists(save_dir):
    #             os.makedirs(save_dir, exist_ok=True)
    #
    #         result_img = PILImage.fromarray(combined_img)
    #         result_img.save(save_path)
    #         print(f"✅ 组合图片已保存：{save_path}")
    #         print(f"   分辨率：{total_w}x{total_h} 像素")
    #
    #     return save_path
    @staticmethod
    def combine_images_to_grid(img_list, rows=None, cols=None,
                               save_path='combined_mesh_grid.png',
                               margin=10, bg_color=(255, 255, 255)):
        """
        将多个图片按网格布局组合成一张大图

        Args:
            img_list: 图片列表，每个元素是 numpy array (H, W, 3) 或 (H, W, 4)
            rows: 行数，None 则自动计算
            cols: 列数，None 则自动计算
            save_path: 保存路径
            margin: 图片之间的边距（像素）
            bg_color: 背景颜色 (R, G, B)

        Returns:
            组合后的图片路径
        """
        n_imgs = len(img_list)
        if n_imgs == 0:
            print("⚠️ 图片列表为空")
            return None

        # 过滤掉无效图片（尺寸为0或None）
        valid_imgs = []
        for idx, img in enumerate(img_list):
            if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print(f"⚠️ 跳过第 {idx+1} 张无效图片")
                continue
            valid_imgs.append(img)

        if len(valid_imgs) == 0:
            print("⚠️ 没有有效图片可以组合")
            return None

        img_list = valid_imgs
        n_imgs = len(img_list)

        # 自动计算行列
        if rows is None and cols is None:
            cols = int(np.ceil(np.sqrt(n_imgs)))
            rows = int(np.ceil(n_imgs / cols))
        elif rows is None:
            rows = int(np.ceil(n_imgs / cols))
        elif cols is None:
            cols = int(np.ceil(n_imgs / rows))

        print(f"📊 图片网格布局：{rows}行 × {cols}列，共 {n_imgs} 张图片")

        # 统一图片尺寸（使用最大尺寸）
        max_h = max(img.shape[0] for img in img_list)
        max_w = max(img.shape[1] for img in img_list)

        # 确保尺寸有效
        if max_h <= 0 or max_w <= 0:
            print("⚠️ 图片尺寸无效")
            return None

        print(f"   统一尺寸：{max_w} x {max_h}")

        # 创建空白画布
        total_h = rows * max_h + (rows + 1) * margin
        total_w = cols * max_w + (cols + 1) * margin

        combined_img = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

        # 粘贴每张图片
        for idx, img in enumerate(img_list):
            row = idx // cols
            col = idx % cols

            # 计算位置
            y_pos = margin + row * (max_h + margin)
            x_pos = margin + col * (max_w + margin)

            # 调整图片尺寸
            if img.shape[0] != max_h or img.shape[1] != max_w:
                try:
                    img_pil = PILImage.fromarray(img)
                    img_resized = img_pil.resize((max_w, max_h), PILImage.LANCZOS)
                    img = np.array(img_resized)
                except Exception as e:
                    print(f"⚠️ 第 {idx+1} 张图片 resize 失败：{e}，跳过")
                    continue

            # 处理 alpha 通道
            if img.shape[2] == 4:
                img_rgb = img[:, :, :3]
                alpha = img[:, :, 3:] / 255.0
                img_final = (img_rgb * alpha + bg_color * (1 - alpha)).astype(np.uint8)
            else:
                img_final = img

            # 再次检查尺寸匹配
            if img_final.shape[0] != max_h or img_final.shape[1] != max_w:
                print(f"⚠️ 第 {idx+1} 张图片尺寸不匹配，跳过")
                continue

            # 粘贴到画布
            try:
                combined_img[y_pos:y_pos+max_h, x_pos:x_pos+max_w] = img_final
                print(f"  [{idx + 1}/{n_imgs}] 已放置到位置 ({row}, {col})")
            except Exception as e:
                print(f"⚠️ 第 {idx+1} 张图片粘贴失败：{e}")
                continue

        # 保存图片
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            result_img = PILImage.fromarray(combined_img)
            result_img.save(save_path)
            print(f"✅ 组合图片已保存：{save_path}")
            print(f"   分辨率：{total_w}x{total_h} 像素")

        return save_path


    def vis_mesh_grid_separate_render(mesh_list, titles=None, rows=None, cols=None,
                                      window_size=[800, 600], background='white',
                                      show_edges=False, view='xy',
                                      elevation=None, azimuth=None,
                                      view_vector=None,
                                      save_path='mesh_grid_combined.png',
                                      margin=10,
                                      enable_lighting=True,
                                      anti_aliasing=True,
                                      point_size=5.0,
                                      line_width=2.0,
                                      multi_samples=8):
        """
        将 mesh_list 中的每个 mesh 单独渲染后整合成一张大图

        Args:
            mesh_list: mesh 列表
            titles: 标题列表，与 mesh_list 对应
            rows: 行数
            cols: 列数
            window_size: 单个 mesh 渲染窗口大小 [width, height]
            background: 背景颜色
            show_edges: 是否显示边
            view: 预设视角
            elevation: 仰角
            azimuth: 方位角
            view_vector: 自定义视角向量
            save_path: 保存路径
            margin: 图片间距（像素）

        Returns:
            保存路径
        """
        n_meshes = len(mesh_list)
        if n_meshes == 0:
            print("⚠️ mesh_list 为空")
            return None

        # if titles is None:
        #     titles = [f"Mesh_{i+1}" for i in range(n_meshes)]

        print(f"\n{'='*60}")
        print(f"开始单独渲染 {n_meshes} 个 mesh")
        print(f"{'='*60}")

        # 单独渲染每个 mesh
        img_list = []
        for idx, mesh in enumerate(mesh_list):
            # title = titles[idx] if idx < len(titles) else f"Mesh_{idx+1}"
            # print(f"\n渲染 [{idx + 1}/{n_meshes}]: {title}")
            title = titles[idx] if titles is not None and idx < len(titles) else None

            img = TMCVisPV.render_mesh_to_image(
                mesh,
                title=title,
                window_size=window_size,
                background=background,
                show_edges=show_edges,
                view=view,
                elevation=elevation,
                azimuth=azimuth,
                view_vector=view_vector,
                anti_aliasing=anti_aliasing,
                multi_samples=multi_samples,
                point_size=point_size,
                line_width=line_width,
                lighting=enable_lighting
            )
            img_list.append(img)

        print(f"\n{'='*60}")
        print(f"所有 mesh 渲染完成，开始组合图片")
        print(f"{'='*60}\n")

        # 组合图片
        result_path = TMCVisPV.combine_images_to_grid(
            img_list,
            rows=rows,
            cols=cols,
            save_path=save_path,
            margin=margin,
            bg_color=(255, 255, 255) if background == 'white' else (0, 0, 0)
        )

        return result_path



