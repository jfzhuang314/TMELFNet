import pyvista as pv
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import json
import trimesh

import subprocess
import os

# ======================== 颜色调色板 ========================
LABEL15_PALETTE = np.array([
    [0.9, 0.9, 0.9],      # 0 - 背景
    [0.6, 0.3, 0.0],      # 1
    [0.6, 0.6, 0.0],      # 2
    [0.3, 0.6, 0.0],      # 3
    [0.0, 0.6, 0.6],      # 4
    [0.0, 0.0, 0.6],      # 5
    [0.6, 0.0, 0.6],      # 6
    [0.6, 0.0, 0.3],      # 7
    [0.5, 0.25, 0.0],     # 8
    [0.3, 0.3, 0.0],      # 9
    [0.15, 0.3, 0.0],     # 10
    [0.0, 0.3, 0.3],      # 11
    [0.0, 0.0, 0.3],      # 12
    [0.3, 0.0, 0.3],      # 13
    [0.3, 0.0, 0.15]      # 14
])

DEFAULT_PALETTE = np.array([
    [128, 128, 128],
    [153, 76, 0],
    [153, 153, 0],
    [76, 153, 0],
    [0, 153, 153],
    [0, 0, 153],
    [153, 0, 153],
    [153, 0, 76],
    [64, 64, 64],
    [255, 128, 0],
    [153, 153, 0],
    [76, 153, 0],
    [0, 153, 153],
    [0, 0, 153],
    [153, 0, 153],
    [153, 0, 76],
    [64, 64, 64],
]) / 255.0


def np_to_pcd(arr, color=[1, 0, 0], offset=[0,0,0], point_size=5.0):
    """
    将 NumPy 数组转换为 PyVista 点云
    Args:
        arr: 点云数组，形状为 (N, 3) 或 (N, 6) [包含法线]
        color: RGB 颜色，范围 [0, 1]
        point_size: 点的大小
    """
    arr = np.array(arr)
    points = arr[:, :3]

    # 创建 PolyData
    pcd = pv.PolyData(points)
    # pcd.translate(offset)
    pcd.points = pcd.points + np.array(offset)

    # 设置颜色
    colors = np.tile(color, (len(points), 1))
    pcd.point_data['colors'] = colors

    # 如果有法线
    if arr.shape[1] >= 6:
        pcd.point_data['normals'] = arr[:, 3:6]

    return pcd


def save_pcd(path, pcd):
    """保存点云到文件"""
    pcd.save(path)


def np_to_mesh(vertices, faces, color=[0.5, 0.5, 0.5]):
    """
    将顶点和面数据转换为 PyVista 网格
    Args:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面索引数组，形状为 (M, 3)
        color: 网格颜色，默认为灰色
    """
    vertices = np.array(vertices)
    faces = np.array(faces)

    # PyVista 要求 faces 格式为 [3, v1, v2, v3, 3, v4, v5, v6, ...]
    if faces.shape[1] == 3:
        n_faces = len(faces)
        pv_faces = np.hstack([np.full((n_faces, 1), 3), faces]).flatten()
    else:
        pv_faces = faces

    mesh = pv.PolyData(vertices, pv_faces)
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

    # 设置颜色
    colors = np.tile(color, (len(vertices), 1))
    mesh.point_data['colors'] = colors

    return mesh


def save_mesh(path, mesh):
    """保存网格到文件"""
    mesh.save(path)


def load_mesh(mesh_path, only_tooth_crop=False):
    """加载网格文件"""
    mesh = pv.read(mesh_path)
    mesh.clean(inplace=True)  # 移除重复顶点和退化面

    if only_tooth_crop:
        # 提取最大连通分量
        mesh = mesh.connectivity(extraction_mode='largest')

    return mesh


def get_colored_mesh(mesh, label_arr, palette_type='label15'):
    """
    根据标签数组给 Mesh 上色
    Args:
        mesh: PyVista PolyData 对象
        label_arr: 标签数组，长度等于顶点数
        palette_type: 'label15' 或 'default'
    """
    mesh = mesh.copy()

    if palette_type == 'label15':
        palette = LABEL15_PALETTE.copy()
    else:
        palette = DEFAULT_PALETTE.copy()
        palette[9:] *= 0.4

    label_arr = np.array(label_arr).copy()
    label_arr %= len(palette)

    # 为每个标签分配颜色
    label_colors = np.zeros((len(label_arr), 3))
    for idx in range(len(palette)):
        mask = label_arr == idx
        label_colors[mask] = palette[idx]

    mesh.point_data['labels'] = label_arr
    mesh.point_data['colors'] = label_colors

    return mesh


def np_to_pcd_with_label(arr, label_arr=None, axis=3):
    """
    将带标签的 NumPy 数组转换为彩色点云
    Args:
        arr: 点云数组 (N, D)，D >= 3
        label_arr: 标签数组 (N,)，如果为 None 则从 arr[:, axis] 提取
        axis: 标签所在的轴
    """
    arr = np.array(arr).copy()
    points = arr[:, :3]

    if label_arr is not None:
        arr = np.concatenate([arr[:, :3], label_arr.reshape(-1, 1)], axis=1)

    pcd = pv.PolyData(points)

    # 设置调色板
    palette = np.array([
        [255, 153, 153],
        [153, 76, 0],
        [153, 153, 0],
        [76, 153, 0],
        [0, 153, 153],
        [0, 0, 153],
        [153, 0, 153],
        [153, 0, 76],
        [64, 64, 64],
        [255, 128, 0],
        [153, 153, 0],
        [76, 153, 0],
        [0, 153, 153],
        [0, 0, 153],
        [153, 0, 153],
        [153, 0, 76],
        [64, 64, 64],
    ]) / 255.0
    palette[9:] *= 0.4

    arr[:, axis] %= len(palette)
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palette):
        label_colors[arr[:, axis] == idx] = palte_color

    pcd.point_data['colors'] = label_colors
    pcd.point_data['labels'] = arr[:, axis].astype(int)

    return pcd


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))


def torch_to_numpy(cuda_arr):
    """PyTorch Tensor 转 NumPy"""
    return cuda_arr.cpu().detach().numpy()


def save_np(arr, path):
    """保存 NumPy 数组"""
    with open(path, 'wb') as f:
        np.save(f, arr)


def load_np(path):
    """加载 NumPy 数组"""
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr


def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, "r") as st_json:
        return json.load(st_json)


def save_json(file_path, json_obj):
    """保存 JSON 文件"""
    with open(file_path, "w") as json_file:
        json.dump(json_obj, json_file)


def make_coord_frame(size=1):
    """创建坐标轴"""
    return pv.Axes(show_actor=True, actor_scale=size, line_width=2)


# ======================== 可视化核心函数 ========================

def print_3d(*data_3d_ls, show_edges=False, window_size=[1200, 900], background='white'):
    """
    快速 3D 可视化（交互式）
    Args:
        *data_3d_ls: 任意数量的点云或网格对象
        show_edges: 是否显示网格边
        window_size: 窗口大小 [width, height]
        background: 背景颜色 'white' 或 'black'
    """
    plotter = pv.Plotter(window_size=window_size, off_screen=False)
    plotter.background_color = background

    for idx, item in enumerate(data_3d_ls):
        if isinstance(item, np.ndarray):
            # 自动判断是点云还是 Mesh
            if item.ndim == 2 and item.shape[1] >= 3:
                item = np_to_pcd(item)
            else:
                item = pv.PolyData(item)

        if isinstance(item, pv.DataSet):
            # 如果有颜色数据，使用它
            if 'colors' in item.point_data:
                plotter.add_mesh(
                    item,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    smooth_shading=True if isinstance(item, pv.PolyData) else False,
                    point_size=3.0 if item.n_cells == 0 else None
                )
            elif 'labels' in item.point_data:
                plotter.add_mesh(
                    item,
                    scalars='labels',
                    cmap='tab20',
                    show_edges=show_edges,
                    smooth_shading=True if isinstance(item, pv.PolyData) else False,
                    point_size=3.0 if item.n_cells == 0 else None
                )
            else:
                plotter.add_mesh(
                    item,
                    show_edges=show_edges,
                    smooth_shading=True if isinstance(item, pv.PolyData) else False,
                    point_size=3.0 if item.n_cells == 0 else None
                )

    plotter.show_axes()
    plotter.show()


def print_3d_args(*data_3d_ls, show_edges=False, background='white',
                  save_path=None, window_size=[1920, 1080], view='xy',
                  off_screen=True, dpi=300,
                  view_vector=None, elevation=None, azimuth=None):
    """
    高级 3D 可视化（支持截图保存）
    Args:
        *data_3d_ls: 数据对象列表
        show_edges: 显示边
        background: 背景颜色
        save_path: 保存路径（.png/.jpg/.pdf/.eps）
        window_size: 窗口分辨率
        view: 视角 'iso'（等距）、'xy'、'xz'、'yz'、'default'
        view_vector: 自定义视角向量 [x,y,z]
        elevation: 仰角（度）
        azimuth: 方位角（度）
    """
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.background_color = background

    for item in data_3d_ls:
        if isinstance(item, np.ndarray):
            item = np_to_pcd(item)

        if isinstance(item, pv.DataSet):
            if 'colors' in item.point_data:
                plotter.add_mesh(
                    item,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    smooth_shading=True,
                    point_size=3.0 if item.n_cells == 0 else None,
                    specular=0.3,
                    specular_power=15
                )
            elif 'labels' in item.point_data:
                plotter.add_mesh(
                    item,
                    scalars='labels',
                    cmap='tab20',
                    show_edges=show_edges,
                    smooth_shading=True,
                    point_size=3.0 if item.n_cells == 0 else None,
                    specular=0.3,
                    specular_power=15
                )
            else:
                plotter.add_mesh(
                    item,
                    show_edges=show_edges,
                    smooth_shading=True,
                    point_size=3.0 if item.n_cells == 0 else None,
                    specular=0.3,
                    specular_power=15
                )

    # 设置视角
    if view_vector is not None:
        plotter.view_vector(view_vector)
    elif elevation is not None and azimuth is not None:
        plotter.view_isometric()
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth
        plotter.camera.zoom(1.0)
    elif view == 'xy':
        plotter.view_xy()
    elif view == 'xz':
        plotter.view_xz()
    elif view == 'yz':
        plotter.view_yz()

    plotter.hide_axes()

    # 保存或显示
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if save_path.endswith('.pdf') or save_path.endswith('.eps'):
            plotter.save_graphic(save_path)
        else:
            plotter.screenshot(save_path)
        print(f"✅ 已保存：{save_path}")
    else:
        plotter.show()

    plotter.close()


def vis_mesh_grid(mesh_list, rows=None, cols=None, x_offset=1.2, y_offset=1.2,
                  title_prefix="Mesh", save_path="vis_mesh_grid.png",
                  show_edges=False, resolution=[3840, 2160]):
    """
    网格布局可视化多个 Mesh
    Args:
        mesh_list: Mesh 或点云列表
        rows/cols: 网格行列数
        x_offset/y_offset: 网格间距
        title_prefix: 标题前缀
        save_path: 保存路径
    """
    n_meshes = len(mesh_list)
    if n_meshes == 0:
        print("⚠️ mesh_list 为空")
        return

    # 自动计算行列
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_meshes)))
        rows = int(np.ceil(n_meshes / cols))
    elif rows is None:
        rows = int(np.ceil(n_meshes / cols))
    elif cols is None:
        cols = int(np.ceil(n_meshes / rows))

    print(f"📊 网格布局：{rows}行 × {cols}列，共 {n_meshes} 个 mesh")

    # 创建子图
    plotter = pv.Plotter(shape=(rows, cols), off_screen=True, window_size=resolution)
    plotter.background_color = 'white'

    for idx, mesh_item in enumerate(mesh_list):
        row = idx // cols
        col = idx % cols

        plotter.subplot(row, col)

        if isinstance(mesh_item, np.ndarray):
            mesh_item = np_to_pcd(mesh_item)

        if isinstance(mesh_item, pv.DataSet):
            if 'colors' in mesh_item.point_data:
                plotter.add_mesh(
                    mesh_item,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15
                )
            else:
                plotter.add_mesh(
                    mesh_item,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15
                )

        # 添加标题
        plotter.add_text(f"{title_prefix}_{idx + 1}", position='upper_edge', font_size=12)
        plotter.hide_axes()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if save_path.endswith('.pdf') or save_path.endswith('.eps'):
            plotter.save_graphic(save_path)
        else:
            plotter.screenshot(save_path)
        print(f"✅ 网格图已保存：{save_path}")

    plotter.close()


def vis_mesh_grid_with_titles(mesh_list, titles=None, rows=None, cols=None,
                              x_offset=1.2, y_offset=1.2, save_path="vis_mesh_grid.png",
                              show_edges=False, resolution=[3840, 2160]):
    """
    带自定义标题的网格布局可视化
    Args:
        mesh_list: Mesh 列表
        titles: 自定义标题列表
    """
    n_meshes = len(mesh_list)
    if titles is None:
        titles = [f"Mesh_{i+1}" for i in range(n_meshes)]

    # 自动计算行列
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_meshes)))
        rows = int(np.ceil(n_meshes / cols))
    elif rows is None:
        rows = int(np.ceil(n_meshes / cols))
    elif cols is None:
        cols = int(np.ceil(n_meshes / rows))

    print(f"📊 网格布局：{rows}行 × {cols}列，共 {n_meshes} 个 mesh")

    # 创建子图
    plotter = pv.Plotter(shape=(rows, cols), off_screen=True, window_size=resolution)
    plotter.background_color = 'white'

    for idx, mesh_item in enumerate(mesh_list):
        row = idx // cols
        col = idx % cols

        plotter.subplot(row, col)

        if isinstance(mesh_item, np.ndarray):
            mesh_item = np_to_pcd(mesh_item)

        if isinstance(mesh_item, pv.DataSet):
            if 'colors' in mesh_item.point_data:
                plotter.add_mesh(
                    mesh_item,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15
                )
            elif 'labels' in mesh_item.point_data:
                plotter.add_mesh(
                    mesh_item,
                    scalars='labels',
                    cmap='tab20',
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15
                )
            else:
                plotter.add_mesh(
                    mesh_item,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15
                )

        # 添加自定义标题
        if idx < len(titles):
            plotter.add_text(titles[idx], position='upper_edge', font_size=14)

        plotter.hide_axes()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if save_path.endswith('.pdf') or save_path.endswith('.eps'):
            plotter.save_graphic(save_path)
        else:
            plotter.screenshot(save_path)
        print(f"✅ 网格图已保存：{save_path}")

    plotter.close()



def enhance_pdf_dpi(input_pdf, output_pdf, dpi=600):
    """
    使用 Ghostscript 提升 PDF 中嵌入图像的 DPI
    :param input_pdf: 输入 PDF 路径
    :param output_pdf: 输出 PDF 路径
    :param dpi: 目标 DPI
    """
    gs_command = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/prepress",  # 最高质量设置
        f"-dColorImageResolution={dpi}",
        f"-dGrayImageResolution={dpi}",
        f"-dMonoImageResolution={dpi}",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dMonoImageDownsampleType=/Bicubic",
        "-dAutoRotatePages=/None",
        "-dNOPAUSE",
        "-dBATCH",
        f"-sOutputFile={output_pdf}",
        input_pdf
    ]

    try:
        subprocess.run(gs_command, check=True)
        print(f"✅ PDF 已优化至 {dpi} DPI: {output_pdf}")
    except FileNotFoundError:
        print("❌ 未找到 Ghostscript，请先安装")
        print("   Windows: https://www.ghostscript.com/download/gsdnld.html")
        print("   或使用: choco install ghostscript")
    except Exception as e:
        print(f"❌ 处理失败: {e}")

# # 使用示例
# enhance_pdf_dpi("output.pdf", "output_600dpi.pdf", dpi=600)

def save_high_quality_graphic(plotter, save_path, dpi=600, use_ghostscript=True):
    """
    保存高质量图形，支持 PDF/SVG/EPS
    :param plotter: PyVista Plotter 对象
    :param save_path: 保存路径
    :param dpi: 目标 DPI
    :param use_ghostscript: 是否使用 Ghostscript 后处理（仅 PDF）
    """
    temp_path = None

    if use_ghostscript and save_path.endswith('.pdf'):
        # 先保存临时文件
        temp_path = save_path.replace('.pdf', '_temp.pdf')
        plotter.save_graphic(temp_path)
        plotter.close()

        # 使用 Ghostscript 优化
        enhance_pdf_dpi(temp_path, save_path, dpi)

        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        # 直接保存
        plotter.save_graphic(save_path)
        plotter.close()
        print(f"✅ 矢量图已保存：{save_path}")


def enhance_pdf_dpi(input_pdf, output_pdf, dpi=600):
    """
    使用 Ghostscript 提升 PDF 中嵌入图像的 DPI
    :param input_pdf: 输入 PDF 路径
    :param output_pdf: 输出 PDF 路径
    :param dpi: 目标 DPI
    """
    import subprocess

    # 尝试常见的 Ghostscript 命令
    gs_commands = ["gs", "gswin64c", "gswin32c"]
    gs_cmd = None

    for cmd in gs_commands:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            gs_cmd = cmd
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    if gs_cmd is None:
        print("⚠️ 未找到 Ghostscript，使用默认保存（可能低于目标 DPI）")
        print("   建议安装 Ghostscript 以获得更高质量的 PDF 输出")
        if os.path.exists(input_pdf):
            os.rename(input_pdf, output_pdf)
        return

    gs_command = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/prepress",
        f"-dColorImageResolution={dpi}",
        f"-dGrayImageResolution={dpi}",
        f"-dMonoImageResolution={dpi}",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dMonoImageDownsampleType=/Bicubic",
        "-dAutoRotatePages=/None",
        "-dNOPAUSE",
        "-dBATCH",
        f"-sOutputFile={output_pdf}",
        input_pdf
    ]

    try:
        subprocess.run(gs_command, check=True, capture_output=True)
        print(f"✅ PDF 已优化至 {dpi} DPI: {output_pdf}")
    except Exception as e:
        print(f"⚠️ Ghostscript 处理失败: {e}")
        if os.path.exists(input_pdf):
            os.rename(input_pdf, output_pdf)



class DataPrinter:
    """数据打印类，用于打印data对象的属性信息"""

    def __init__(self, data):
        self.data = data

    def print(self):
        """打印data对象的所有属性信息

        Args:
            data: 数据对象，通常包含pos等属性

        Returns:
            None
        """
        print("=" * 50)
        print("Data Object Attributes:")
        print("=" * 50)

        # 遍历data对象的所有属性
        for attr_name in dir(self.data):
            # 跳过私有属性和方法
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(self.data, attr_name)
                    # 只处理非方法的属性
                    if not callable(attr_value):
                        # 根据属性类型输出不同信息
                        if isinstance(attr_value, (int, float, bool)):  # scalar类型
                            print(f"{attr_name}: {attr_value}")
                        elif isinstance(attr_value, torch.Tensor):  # tensor类型
                            print(f"{attr_name}: Tensor with shape {attr_value.shape}")
                        elif isinstance(attr_value, str):  # 字符串类型
                            print(f"{attr_name}: {attr_value}")
                        elif isinstance(attr_value, (list, tuple)):  # 列表或元组
                            print(f"{attr_name}: {type(attr_value).__name__} with length {len(attr_value)}")
                        elif isinstance(attr_value, np.ndarray):  # numpy数组
                            print(f"{attr_name}: numpy.ndarray with shape {attr_value.shape}")
                        else:  # 其他类型
                            print(f"{attr_name}: {type(attr_value).__name__}")
                except:
                    # 某些属性可能无法访问，跳过
                    continue

        print("\n" + "=" * 50)
        print("Position Information (data.pos):")
        print("=" * 50)

        # 输出data.pos的详细信息
        if hasattr(self.data, 'pos') and self.data.pos is not None:
            pos = self.data.pos

            # 确保pos是numpy数组或tensor
            if isinstance(pos, torch.Tensor):
                pos_array = pos.detach().cpu().numpy()
            else:
                pos_array = np.array(pos)

            # 计算边界框信息
            xmin = np.min(pos_array[:, 0])
            xmax = np.max(pos_array[:, 0])
            ymin = np.min(pos_array[:, 1])
            ymax = np.max(pos_array[:, 1])
            zmin = np.min(pos_array[:, 2])
            zmax = np.max(pos_array[:, 2])

            # 计算中心点坐标
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            center_z = (zmin + zmax) / 2

            print(f"Shape: {pos_array.shape}")
            print(f"X range: [{xmin:.6f}, {xmax:.6f}]")
            print(f"Y range: [{ymin:.6f}, {ymax:.6f}]")
            print(f"Z range: [{zmin:.6f}, {zmax:.6f}]")
            print(f"Center point: ({center_x:.6f}, {center_y:.6f}, {center_z:.6f})")
            print(f"Bounding box size: ({xmax - xmin:.6f}, {ymax - ymin:.6f}, {zmax - zmin:.6f})")
        else:
            print("data.pos attribute not found or is None")

        print("=" * 50)



class DataVisualizer:
    """数据可视化类，基于pv_gen_utils可视化Data对象"""

    def __init__(self, data, use_pv_gen_utils=True):
        """
        初始化可视化器
        :param data: PyG Data对象，包含pos、face、y等属性
        :param use_pv_gen_utils: 是否使用pv_gen_utils库
        """
        self.data = data
        self.use_pv_gen_utils = use_pv_gen_utils

    def _to_numpy(self, tensor):
        """将Tensor转换为NumPy数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def visualize_point_cloud(self, show_edges=False, background='white',
                             window_size=[1200, 900], save_path=None,
                             anti_aliasing=True, multi_samples=8, point_smoothing=True):
        """
        可视化点云数据
        :param show_edges: 是否显示边
        :param background: 背景颜色 'white' 或 'black'
        :param window_size: 窗口大小
        :param save_path: 保存路径（可选）
        :param anti_aliasing: 是否启用抗锯齿
        :param multi_samples: 多重采样数（4, 8, 16），数值越高质量越好但速度越慢
        :param point_smoothing: 点平滑
        """
        if not hasattr(self.data, 'pos'):
            raise AttributeError("Data对象没有pos属性")

        pos = self._to_numpy(self.data.pos)

        # 创建带标签的点云
        if hasattr(self.data, 'y'):
            labels = self._to_numpy(self.data.y).flatten()
            pcd = np_to_pcd_with_label(pos, labels)
        else:
            pcd = np_to_pcd(pos, color=[0.7, 0.7, 0.7])

        if save_path:
            print_3d_args(pcd, show_edges=show_edges, background=background,
                              save_path=save_path, window_size=window_size)
        else:
            # 使用自定义plotter以支持抗锯齿
            plotter = pv.Plotter(window_size=window_size, off_screen=False)
            plotter.background_color = background

            # 启用抗锯齿设置
            if anti_aliasing:
                plotter.render_window.SetMultiSamples(multi_samples)

            # 添加点云
            if 'colors' in pcd.point_data:
                plotter.add_mesh(
                    pcd,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    render_points_as_spheres=point_smoothing,
                    point_size=3.0
                )
            elif 'labels' in pcd.point_data:
                plotter.add_mesh(
                    pcd,
                    scalars='labels',
                    cmap='tab20',
                    show_edges=show_edges,
                    render_points_as_spheres=point_smoothing,
                    point_size=3.0
                )
            else:
                plotter.add_mesh(
                    pcd,
                    show_edges=show_edges,
                    render_points_as_spheres=point_smoothing,
                    point_size=3.0
                )

            plotter.show_axes()
            plotter.show()
            plotter.close()

    # def visualize_mesh(self, show_edges=False, background='white',
    #                   window_size=[1200, 900], save_path=None, use_vertex_color=True,
    #                   anti_aliasing=True, multi_samples=8, line_smoothing=True,
    #                   view_vector=None, elevation=None, azimuth=None, view='xy',
    #                   point_smoothing=True, polygon_smoothing=True):
    #     """
    #     可视化网格数据
    #     :param show_edges: 是否显示边
    #     :param background: 背景颜色
    #     :param window_size: 窗口大小
    #     :param save_path: 保存路径（可选）
    #     :param use_vertex_color: 是否使用顶点着色（基于标签）
    #     :param anti_aliasing: 是否启用抗锯齿
    #     :param multi_samples: 多重采样数（4, 8, 16），数值越高质量越好但速度越慢
    #     :param line_smoothing: 线条平滑
    #     :param point_smoothing: 点平滑
    #     :param polygon_smoothing: 多边形平滑
    #     """
    #     if not hasattr(self.data, 'pos') or not hasattr(self.data, 'face'):
    #         raise AttributeError("Data对象缺少pos或face属性")
    #
    #     vertices = self._to_numpy(self.data.pos)
    #     faces = self._to_numpy(self.data.face).T  # 转置为(M, 3)格式
    #
    #     # 创建基础网格
    #     mesh = np_to_mesh(vertices, faces)
    #
    #     # 如果有标签，使用标签着色
    #     if hasattr(self.data, 'y') and use_vertex_color:
    #         labels = self._to_numpy(self.data.y).flatten()
    #         mesh = get_colored_mesh(mesh, labels, palette_type='label15')
    #
    #     if save_path:
    #         print_3d_args(mesh, show_edges=show_edges, background=background,
    #                           view_vector=view_vector, elevation=elevation, azimuth=azimuth, view=view,
    #                           save_path=save_path, window_size=window_size)
    #     else:
    #         # 使用自定义plotter以支持抗锯齿
    #         plotter = pv.Plotter(window_size=window_size, off_screen=False)
    #         plotter.background_color = background
    #
    #         # 启用抗锯齿设置
    #         if anti_aliasing:
    #             plotter.render_window.SetMultiSamples(multi_samples)
    #             if hasattr(plotter.renderer, 'EnableSSAO'):
    #                 plotter.renderer.EnableSSAO()
    #
    #         # 添加网格
    #         if 'colors' in mesh.point_data:
    #             plotter.add_mesh(
    #                 mesh,
    #                 scalars='colors',
    #                 rgb=True,
    #                 show_edges=show_edges,
    #                 smooth_shading=True,
    #                 specular=0.3,
    #                 specular_power=15,
    #                 lighting=True,
    #                 render_lines_as_tubes=line_smoothing,
    #                 render_points_as_spheres=point_smoothing
    #             )
    #         elif 'labels' in mesh.point_data:
    #             plotter.add_mesh(
    #                 mesh,
    #                 scalars='labels',
    #                 cmap='tab20',
    #                 show_edges=show_edges,
    #                 smooth_shading=True,
    #                 specular=0.3,
    #                 specular_power=15,
    #                 lighting=True,
    #                 render_lines_as_tubes=line_smoothing,
    #                 render_points_as_spheres=point_smoothing
    #             )
    #         else:
    #             plotter.add_mesh(
    #                 mesh,
    #                 show_edges=show_edges,
    #                 smooth_shading=True,
    #                 specular=0.3,
    #                 specular_power=15,
    #                 lighting=True,
    #                 render_lines_as_tubes=line_smoothing,
    #                 render_points_as_spheres=point_smoothing
    #             )
    #
    #         # 启用额外的渲染优化
    #         if polygon_smoothing:
    #             plotter.renderer.UseShading = True
    #
    #         # 设置视角
    #         if view_vector is not None:
    #             plotter.view_vector(view_vector)
    #         elif elevation is not None and azimuth is not None:
    #             plotter.view_isometric()
    #             plotter.camera.elevation = elevation
    #             plotter.camera.azimuth = azimuth
    #             plotter.camera.zoom(1.0)
    #         elif view == 'xy':
    #             plotter.view_xy()
    #         elif view == 'xz':
    #             plotter.view_xz()
    #         elif view == 'yz':
    #             plotter.view_yz()
    #
    #         plotter.show_axes()
    #         plotter.show()
    #         plotter.close()
    def visualize_mesh(self, show_edges=False, background='white',
                      window_size=[1200, 900], save_path=None, use_vertex_color=True,
                      anti_aliasing=True, multi_samples=8, line_smoothing=True,
                      view_vector=None, elevation=None, azimuth=None, view='xy',
                      point_smoothing=True, polygon_smoothing=True,
                      dpi=600):
        """
        可视化网格数据
        :param show_edges: 是否显示边
        :param background: 背景颜色
        :param window_size: 窗口大小
        :param save_path: 保存路径（可选），支持 .png, .jpg, .svg, .pdf
        :param use_vertex_color: 是否使用顶点着色（基于标签）
        :param anti_aliasing: 是否启用抗锯齿
        :param multi_samples: 多重采样数（4, 8, 16），数值越高质量越好但速度越慢
        :param line_smoothing: 线条平滑
        :param point_smoothing: 点平滑
        :param polygon_smoothing: 多边形平滑
        :param view_vector: 自定义视角向量
        :param elevation: 仰角
        :param azimuth: 方位角
        :param view: 预设视角
        :param dpi: 输出 DPI（对 PNG/JPG 通过窗口缩放实现，对 PDF/SVG 通过提高渲染分辨率）
        """
        if not hasattr(self.data, 'pos') or not hasattr(self.data, 'face'):
            raise AttributeError("Data对象缺少pos或face属性")

        vertices = self._to_numpy(self.data.pos)
        faces = self._to_numpy(self.data.face).T  # 转置为(M, 3)格式

        # 创建基础网格
        mesh = np_to_mesh(vertices, faces)

        # 如果有标签，使用标签着色
        if hasattr(self.data, 'y') and use_vertex_color:
            labels = self._to_numpy(self.data.y).flatten()
            mesh = get_colored_mesh(mesh, labels, palette_type='label15')

        if save_path:
            # 根据文件类型选择保存方式
            if save_path.endswith(('.svg', '.pdf', '.eps')):
                # 矢量格式：提高窗口分辨率以确保嵌入元素的高质量
                scale_factor = dpi / 300
                high_res_window = [int(w * scale_factor) for w in window_size]

                plotter = pv.Plotter(window_size=high_res_window, off_screen=True)
                plotter.background_color = background

                # 启用抗锯齿
                if anti_aliasing:
                    plotter.render_window.SetMultiSamples(multi_samples)

                if 'colors' in mesh.point_data:
                    plotter.add_mesh(
                        mesh, scalars='colors', rgb=True, show_edges=show_edges,
                        smooth_shading=True, specular=0.3, specular_power=15
                    )
                else:
                    plotter.add_mesh(
                        mesh, show_edges=show_edges,
                        smooth_shading=True, specular=0.3, specular_power=15
                    )

                # 设置视角
                if view_vector is not None:
                    plotter.view_vector(view_vector)
                elif elevation is not None and azimuth is not None:
                    plotter.view_isometric()
                    plotter.camera.elevation = elevation
                    plotter.camera.azimuth = azimuth
                elif view == 'xy':
                    plotter.view_xy()
                elif view == 'xz':
                    plotter.view_xz()
                elif view == 'yz':
                    plotter.view_yz()

                plotter.hide_axes()
                plotter.save_graphic(save_path)
                print(f"✅ 矢量图已保存（{dpi} DPI 等效）：{save_path}")
                plotter.close()
            else:
                # 位图格式：使用高分辨率窗口
                print_3d_args(mesh, show_edges=show_edges, background=background,
                              view_vector=view_vector, elevation=elevation, azimuth=azimuth, view=view,
                              save_path=save_path, window_size=window_size, dpi=dpi)
        else:
            # 使用自定义plotter以支持抗锯齿
            plotter = pv.Plotter(window_size=window_size, off_screen=False)
            plotter.background_color = background

            # 启用抗锯齿设置
            if anti_aliasing:
                plotter.render_window.SetMultiSamples(multi_samples)
                if hasattr(plotter.renderer, 'EnableSSAO'):
                    plotter.renderer.EnableSSAO()

            # 添加网格
            if 'colors' in mesh.point_data:
                plotter.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15,
                    lighting=True,
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
                    specular=0.3,
                    specular_power=15,
                    lighting=True,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )
            else:
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15,
                    lighting=True,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )

            # 启用额外的渲染优化
            if polygon_smoothing:
                plotter.renderer.UseShading = True

            # 设置视角
            if view_vector is not None:
                plotter.view_vector(view_vector)
            elif elevation is not None and azimuth is not None:
                plotter.view_isometric()
                plotter.camera.elevation = elevation
                plotter.camera.azimuth = azimuth
                plotter.camera.zoom(1.0)
            elif view == 'xy':
                plotter.view_xy()
            elif view == 'xz':
                plotter.view_xz()
            elif view == 'yz':
                plotter.view_yz()

            plotter.show_axes()
            plotter.show()
            plotter.close()

    def visualize_mesh_with_instances(self, show_edges=False, background='white',
                                     window_size=[1200, 900], save_path=None,
                                     anti_aliasing=True, multi_samples=8,
                                     line_smoothing=True, point_smoothing=True):
        """
        可视化带实例信息的网格
        :param show_edges: 是否显示边
        :param background: 背景颜色
        :param window_size: 窗口大小
        :param save_path: 保存路径（可选）
        :param anti_aliasing: 是否启用抗锯齿
        :param multi_samples: 多重采样数
        :param line_smoothing: 线条平滑
        :param point_smoothing: 点平滑
        """
        if not hasattr(self.data, 'pos') or not hasattr(self.data, 'face'):
            raise AttributeError("Data对象缺少pos或face属性")

        vertices = self._to_numpy(self.data.pos)
        faces = self._to_numpy(self.data.face).T

        # 创建网格
        mesh = np_to_mesh(vertices, faces)

        # 使用实例ID着色
        if hasattr(self.data, 'instances'):
            instances = self._to_numpy(self.data.instances).flatten()
            mesh = get_colored_mesh(mesh, instances, palette_type='default')
        elif hasattr(self.data, 'y'):
            labels = self._to_numpy(self.data.y).flatten()
            mesh = get_colored_mesh(mesh, labels, palette_type='label15')

        if save_path:
            print_3d_args(mesh, show_edges=show_edges, background=background,
                              save_path=save_path, window_size=window_size)
        else:
            # 使用自定义plotter以支持抗锯齿
            plotter = pv.Plotter(window_size=window_size, off_screen=False)
            plotter.background_color = background

            # 启用抗锯齿设置
            if anti_aliasing:
                plotter.render_window.SetMultiSamples(multi_samples)

            # 添加网格
            if 'colors' in mesh.point_data:
                plotter.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15,
                    lighting=True,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )
            else:
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=15,
                    lighting=True,
                    render_lines_as_tubes=line_smoothing,
                    render_points_as_spheres=point_smoothing
                )

            plotter.show_axes()
            plotter.show()
            plotter.close()

    def visualize_multi_view(self, views=['xy', 'xz', 'yz', 'iso'],
                            background='white', window_size=[1920, 1080],
                            save_dir='./vis_output', filename='multi_view',
                            anti_aliasing=True, multi_samples=8):
        """
        多视角可视化并保存
        :param views: 视角列表 ['xy', 'xz', 'yz', 'iso']
        :param background: 背景颜色
        :param window_size: 窗口大小
        :param save_dir: 保存目录
        :param filename: 文件名前缀
        :param anti_aliasing: 是否启用抗锯齿
        :param multi_samples: 多重采样数
        """
        os.makedirs(save_dir, exist_ok=True)

        for view in views:
            save_path = os.path.join(save_dir, f"{filename}_{view}.png")
            try:
                self.visualize_mesh(background=background, window_size=window_size,
                                   save_path=save_path, anti_aliasing=anti_aliasing,
                                   multi_samples=multi_samples)
                print(f"✅ {view}视角已保存: {save_path}")
            except Exception as e:
                print(f"❌ {view}视角保存失败: {e}")

    def visualize_comparison(self, data_list, titles=None, background='white',
                           window_size=[3840, 2160], save_path=None,
                           anti_aliasing=True, multi_samples=8):
        """
        对比可视化多个Data对象
        :param data_list: Data对象列表
        :param titles: 标题列表
        :param background: 背景颜色
        :param window_size: 窗口大小
        :param save_path: 保存路径
        :param anti_aliasing: 是否启用抗锯齿
        :param multi_samples: 多重采样数
        """
        meshes = []
        for i, data_item in enumerate(data_list):
            vertices = self._to_numpy(data_item.pos)
            faces = self._to_numpy(data_item.face).T

            mesh = np_to_mesh(vertices, faces)

            if hasattr(data_item, 'y'):
                labels = self._to_numpy(data_item.y).flatten()
                mesh = get_colored_mesh(mesh, labels, palette_type='label15')

            meshes.append(mesh)

        if titles is None:
            titles = [f"Data_{i+1}" for i in range(len(data_list))]

        if save_path:
            vis_mesh_grid_with_titles(
                meshes, titles=titles, save_path=save_path,
                show_edges=False, resolution=window_size
            )
        else:
            vis_mesh_grid(meshes, save_path=None, show_edges=False)

    def quick_vis(self, mode='auto', **kwargs):
        """
        快速可视化（自动检测数据类型）
        :param mode: 'auto', 'point', 'mesh', 'instance'
        :param kwargs: 传递给具体可视化函数的参数
                       常用参数:
                       - anti_aliasing: True/False 是否抗锯齿
                       - multi_samples: 4/8/16 抗锯齿质量
                       - background: 'white'/'black' 背景色
                       - window_size: [width, height] 窗口大小
                       - show_edges: True/False 显示边
                       - line_smoothing: True/False 线条平滑
                       - point_smoothing: True/False 点平滑
                       - polygon_smoothing: True/False 多边形平滑
        """
        if mode == 'auto':
            if hasattr(self.data, 'face'):
                mode = 'mesh'
            else:
                mode = 'point'

        if mode == 'point':
            self.visualize_point_cloud(**kwargs)
        elif mode == 'mesh':
            self.visualize_mesh(**kwargs)
        elif mode == 'instance':
            self.visualize_mesh_with_instances(**kwargs)
        else:
            raise ValueError(f"不支持的mode: {mode}")



# class DataVisualizer:
#     """数据可视化类，基于pv_gen_utils可视化Data对象"""
#
#     def __init__(self, data, use_pv_gen_utils=True):
#         """
#         初始化可视化器
#         :param data: PyG Data对象，包含pos、face、y等属性
#         :param use_pv_gen_utils: 是否使用pv_gen_utils库
#         """
#         self.data = data
#         self.use_pv_gen_utils = use_pv_gen_utils
#
#     def _to_numpy(self, tensor):
#         """将Tensor转换为NumPy数组"""
#         if isinstance(tensor, torch.Tensor):
#             return tensor.detach().cpu().numpy()
#         return np.array(tensor)
#
#     def visualize_point_cloud(self, show_edges=False, background='white',
#                              window_size=[1200, 900], save_path=None):
#         """
#         可视化点云数据
#         :param show_edges: 是否显示边
#         :param background: 背景颜色 'white' 或 'black'
#         :param window_size: 窗口大小
#         :param save_path: 保存路径（可选）
#         """
#         if not hasattr(self.data, 'pos'):
#             raise AttributeError("Data对象没有pos属性")
#
#         pos = self._to_numpy(self.data.pos)
#
#         # 创建带标签的点云
#         if hasattr(self.data, 'y'):
#             labels = self._to_numpy(self.data.y).flatten()
#             pcd = np_to_pcd_with_label(pos, labels)
#         else:
#             pcd = np_to_pcd(pos, color=[0.7, 0.7, 0.7])
#
#         if save_path:
#             print_3d_args(pcd, show_edges=show_edges, background=background,
#                               save_path=save_path, window_size=window_size)
#         else:
#             print_3d(pcd, show_edges=show_edges, background=background,
#                          window_size=window_size)
#
#     def visualize_mesh(self, show_edges=False, background='white',
#                       window_size=[1200, 900], save_path=None, use_vertex_color=True):
#         """
#         可视化网格数据
#         :param show_edges: 是否显示边
#         :param background: 背景颜色
#         :param window_size: 窗口大小
#         :param save_path: 保存路径（可选）
#         :param use_vertex_color: 是否使用顶点着色（基于标签）
#         """
#         if not hasattr(self.data, 'pos') or not hasattr(self.data, 'face'):
#             raise AttributeError("Data对象缺少pos或face属性")
#
#         vertices = self._to_numpy(self.data.pos)
#         faces = self._to_numpy(self.data.face).T  # 转置为(M, 3)格式
#
#         # 创建基础网格
#         mesh = np_to_mesh(vertices, faces)
#
#         # 如果有标签，使用标签着色
#         if hasattr(self.data, 'y') and use_vertex_color:
#             labels = self._to_numpy(self.data.y).flatten()
#             mesh = get_colored_mesh(mesh, labels, palette_type='label15')
#
#         if save_path:
#             print_3d_args(mesh, show_edges=show_edges, background=background,
#                               save_path=save_path, window_size=window_size)
#         else:
#             print_3d(mesh, show_edges=show_edges, background=background,
#                          window_size=window_size)
#
#     def visualize_mesh_with_instances(self, show_edges=False, background='white',
#                                      window_size=[1200, 900], save_path=None):
#         """
#         可视化带实例信息的网格
#         :param show_edges: 是否显示边
#         :param background: 背景颜色
#         :param window_size: 窗口大小
#         :param save_path: 保存路径（可选）
#         """
#         if not hasattr(self.data, 'pos') or not hasattr(self.data, 'face'):
#             raise AttributeError("Data对象缺少pos或face属性")
#
#         vertices = self._to_numpy(self.data.pos)
#         faces = self._to_numpy(self.data.face).T
#
#         # 创建网格
#         mesh = np_to_mesh(vertices, faces)
#
#         # 使用实例ID着色
#         if hasattr(self.data, 'instances'):
#             instances = self._to_numpy(self.data.instances).flatten()
#             mesh = get_colored_mesh(mesh, instances, palette_type='default')
#         elif hasattr(self.data, 'y'):
#             labels = self._to_numpy(self.data.y).flatten()
#             mesh = get_colored_mesh(mesh, labels, palette_type='label15')
#
#         if save_path:
#             print_3d_args(mesh, show_edges=show_edges, background=background,
#                               save_path=save_path, window_size=window_size)
#         else:
#             print_3d(mesh, show_edges=show_edges, background=background,
#                          window_size=window_size)
#
#     def visualize_multi_view(self, views=['xy', 'xz', 'yz', 'iso'],
#                             background='white', window_size=[1920, 1080],
#                             save_dir='./vis_output', filename='multi_view'):
#         """
#         多视角可视化并保存
#         :param views: 视角列表 ['xy', 'xz', 'yz', 'iso']
#         :param background: 背景颜色
#         :param window_size: 窗口大小
#         :param save_dir: 保存目录
#         :param filename: 文件名前缀
#         """
#         os.makedirs(save_dir, exist_ok=True)
#
#         for view in views:
#             save_path = os.path.join(save_dir, f"{filename}_{view}.png")
#             try:
#                 self.visualize_mesh(background=background, window_size=window_size,
#                                    save_path=save_path)
#                 print(f"✅ {view}视角已保存: {save_path}")
#             except Exception as e:
#                 print(f"❌ {view}视角保存失败: {e}")
#
#     def visualize_comparison(self, data_list, titles=None, background='white',
#                            window_size=[3840, 2160], save_path=None):
#         """
#         对比可视化多个Data对象
#         :param data_list: Data对象列表
#         :param titles: 标题列表
#         :param background: 背景颜色
#         :param window_size: 窗口大小
#         :param save_path: 保存路径
#         """
#         meshes = []
#         for i, data_item in enumerate(data_list):
#             vertices = self._to_numpy(data_item.pos)
#             faces = self._to_numpy(data_item.face).T
#
#             mesh = np_to_mesh(vertices, faces)
#
#             if hasattr(data_item, 'y'):
#                 labels = self._to_numpy(data_item.y).flatten()
#                 mesh = get_colored_mesh(mesh, labels, palette_type='label15')
#
#             meshes.append(mesh)
#
#         if titles is None:
#             titles = [f"Data_{i+1}" for i in range(len(data_list))]
#
#         if save_path:
#             vis_mesh_grid_with_titles(
#                 meshes, titles=titles, save_path=save_path,
#                 show_edges=False, resolution=window_size
#             )
#         else:
#             vis_mesh_grid(meshes, save_path=None, show_edges=False)
#
#     def quick_vis(self, mode='auto', **kwargs):
#         """
#         快速可视化（自动检测数据类型）
#         :param mode: 'auto', 'point', 'mesh', 'instance'
#         :param kwargs: 传递给具体可视化函数的参数
#         """
#         if mode == 'auto':
#             if hasattr(self.data, 'face'):
#                 mode = 'mesh'
#             else:
#                 mode = 'point'
#
#         if mode == 'point':
#             self.visualize_point_cloud(**kwargs)
#         elif mode == 'mesh':
#             self.visualize_mesh(**kwargs)
#         elif mode == 'instance':
#             self.visualize_mesh_with_instances(**kwargs)
#         else:
#             raise ValueError(f"不支持的mode: {mode}")
#
