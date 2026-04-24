import os
from copy import deepcopy

import torch
import open3d as o3d
import numpy as np
from typing import Optional, Tuple, Literal, Union

from open3d.cpu.pybind.visualization import MeshColorOption, MeshShadeOption
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform  # 继承自该基类
from overrides import overrides

# ===================== 原生枚举类型定义（映射Open3D原生值）=====================
# 数据渲染模式
DataMode = Literal["point", "mesh", "vol"]
# 颜色模式
ColorMode = Literal["label_color", "edge_color", "uniform_color"]
# 视角模式
ViewMode = Literal["top", "left", "front", "custom"]
# 3维向量类型（兼容list/tuple/np.array/torch.Tensor）
Vec3 = Union[Tuple[float, float, float], list[float], np.ndarray, torch.Tensor]


# ===================== 核心类：RenderAndSaver（继承BaseTransform）=====================
class VisRenderSaver(BaseTransform):
    """
    PyG BaseTransform派生类，用于3D数据渲染与保存
    核心特性：使用Open3D原生枚举/参数，无自定义封装类
    """
    def __init__(
            self,
            # 路径相关
            save_file: bool = False,  # 是否保存渲染结果到文件
            root: str = "./render_output",  # 保存根目录
            classname: str = "default",  # 分类目录
            # 数据渲染模式
            data_mode: DataMode = "point",
            # 颜色模式相关
            color_mode: Optional[ColorMode] = None,
            uniform_color: Tuple[float, float, float] = (0.3, 0.3, 0.3),  # 纯色模式默认色
            palette: Optional[list[Tuple[float, float, float]]] = None,  # 颜色调色板
            # 窗口相关
            window_visible: bool = False,  # 修正拼写错误：window_visable → window_visible
            window_size: Tuple[int, int] = (800, 600),  # 窗口尺寸（int更合理）
            # 视角相关（Open3D原生相机参数）
            view_mode: ViewMode = "top",
            lookat: Vec3 = (0.0, 0.0, 0.0),
            front: Vec3 = (0.0, 0.0, -1.0),
            up: Vec3 = (0.0, 1.0, 0.0),
            zoom: float = 0.5,  # 相机缩放比例
            # -------------------------- Open3D RenderOption 原生参数 --------------------------
            # 基础渲染配置
            point_size: float = 5.0,
            line_width: float = 2.0,
            background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            show_coordinate_frame: bool = True,
            # 点云专属配置
            point_show_normal: bool = False,
            # 网格专属配置（Open3D原生枚举值，注释对应含义）
            mesh_show_wireframe: bool = False,
            mesh_show_back_face: bool = False,
            # Open3D原生mesh_color_option值：0-Default 1-Color 2-Normal 3-Vertex 4-Material
            mesh_color_option: int = 0,
            # Open3D原生mesh_shade_option值：0-Flat 1-Smooth
            mesh_shade_option: int = 0,
            # 光照/阴影配置（Open3D原生）
            light_on: bool = True,
            # 打印信息
            print_info: bool = True,
    ):
        super().__init__()  # 调用BaseTransform构造函数

        # 路径/保存配置
        self.save_file = save_file
        self.root = root
        self.classname = classname
        # 数据/颜色配置
        self.data_mode = data_mode
        self.color_mode = color_mode
        self.uniform_color = uniform_color
        self.palette = palette
        # 窗口配置
        self.window_visible = window_visible
        self.window_size = window_size
        # 视角配置（直接映射Open3D原生参数）
        self.view_mode = view_mode
        self.lookat = lookat
        self.front = front
        self.up = up
        self.zoom = zoom
        # Open3D RenderOption原生参数（直接存储，无封装）
        self.point_size = point_size
        self.line_width = line_width
        self.background_color = background_color
        self.show_coordinate_frame = show_coordinate_frame
        self.point_show_normal = point_show_normal
        self.mesh_show_wireframe = mesh_show_wireframe
        self.mesh_show_back_face = mesh_show_back_face
        self.mesh_color_option = mesh_color_option  # Open3D原生int值
        self.mesh_shade_option = mesh_shade_option  # Open3D原生int值
        self.light_on = light_on
        self.print_info = print_info

        # 预设相机视角（Open3D标准原生参数）
        self.preset_camera = {
            "top": {"lookat": (0.0, 0.0, 0.0), "front": (0.0, 0.0, 1.0), "up": (0.0, 1.0, 0.0)},
            "left": {"lookat": (0.0, 0.0, 0.0), "front": (1.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0)},
            "front": {"lookat": (0.0, 0.0, 0.0), "front": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)},
        }

    def validate_data(self, data: Data):
        """校验Data对象的合法性（核心属性检查）"""
        # 必选：pos属性（N,3）
        if not hasattr(data, "pos") or not isinstance(data.pos, torch.Tensor) or data.pos.ndim != 2 or data.pos.shape[
            1] != 3:
            raise AttributeError(
                f"Data对象必须包含pos属性（形状为(N,3)的torch.Tensor），当前pos: {getattr(data, 'pos', '不存在')}")
        # 必选：filename属性（用于保存文件）
        if self.save_file and (not hasattr(data, "filename") or not isinstance(data.filename, str)):
            raise AttributeError(
                f"save_file=True时，Data对象必须包含str类型的filename属性，当前filename: {getattr(data, 'filename', '不存在')}")
        # mesh模式必选：face属性（3,M）
        if self.data_mode == "mesh" and (
                not hasattr(data, "face") or not isinstance(data.face, torch.Tensor) or data.face.ndim != 2 or
                data.face.shape[0] != 3):
            raise AttributeError(
                f"mesh模式下，Data对象必须包含face属性（形状为(3,M)的torch.Tensor），当前face: {getattr(data, 'face', '不存在')}")
        # vol模式必选：vol_dims属性（3,）
        if self.data_mode == "vol" and (not hasattr(data, "vol_dims") or not isinstance(data.vol_dims,
                                                                                        torch.Tensor) or data.vol_dims.ndim != 1 or
                                        data.vol_dims.shape[0] != 3):
            raise AttributeError(
                f"vol模式下，Data对象必须包含vol_dims属性（形状为(3,)的torch.Tensor），当前vol_dims: {getattr(data, 'vol_dims', '不存在')}")
        # color_mode校验
        if self.color_mode is not None and self.color_mode not in \
                ["label_color", "edge_color", "tte_color", "uniform_color"]:
            raise ValueError(f"color_mode必须为label_color/edge_color/tte_color/uniform_color，当前值：{self.color_mode}")
        # Open3D原生参数范围校验
        if self.mesh_color_option not in [0, 1, 2, 3, 4,9]:
            raise ValueError(f"mesh_color_option必须为0-4,9（Open3D原生值），当前值：{self.mesh_color_option}")
        if self.mesh_shade_option not in [0, 1]:
            raise ValueError(f"mesh_shade_option必须为0/1（Open3D原生值），当前值：{self.mesh_shade_option}")

    def get_main_geometry(self, data: Data):
        """将PyG Data对象转换为Open3D原生几何对象（point/mesh/vol）"""
        pos_np = data.pos.detach().cpu().numpy().astype(np.float64)  # Open3D强制float64

        # 点云模式（Open3D原生PointCloud）
        if self.data_mode == "point":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pos_np)
            # 加载法向量（若有，Open3D原生属性）
            if hasattr(data, "normals") and data.normals.shape == data.pos.shape:
                pcd.normals = o3d.utility.Vector3dVector(data.normals.detach().cpu().numpy().astype(np.float64))
            return pcd

        # 网格模式（Open3D原生TriangleMesh，PyG face(3,M) → Open3D face(M,3)）
        elif self.data_mode == "mesh":
            face_np = data.face.detach().cpu().numpy().astype(np.int32).T
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(pos_np)
            mesh.triangles = o3d.utility.Vector3iVector(face_np)
            mesh.compute_vertex_normals()  # Open3D原生方法，保证着色正常
            return mesh

        # 体素模式（Open3D原生VoxelGrid）
        elif self.data_mode == "vol":
            vol_dims = data.vol_dims.detach().cpu().numpy().astype(int)
            voxel_size = np.max(pos_np) / max(vol_dims) if max(vol_dims) > 0 else 0.1
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pos_np)),
                voxel_size=voxel_size
            )
            return voxel_grid

    def set_geometry_color(self, o3d_geom, data: Data, color_mode: str = "uniform_color"):
        """根据color_mode为Open3D原生几何对象设置颜色"""
        if self.color_mode is None:
            return  # 不设置颜色，使用Open3D默认

        def set_colors(o3d_geom, colors: list):
            # 检查颜色数据格式
            if not isinstance(colors, (list, np.ndarray)):
                raise TypeError("colors must be a list or numpy array")

            # 转换为numpy数组并确保数据类型为float64
            colors_array = np.array(colors, dtype=np.float64)

            # 根据几何对象类型设置颜色
            if isinstance(o3d_geom, o3d.geometry.PointCloud):
                # 点云使用 colors 属性
                o3d_geom.colors = o3d.utility.Vector3dVector(colors_array)
            elif isinstance(o3d_geom, o3d.geometry.TriangleMesh):
                # 网格使用 vertex_colors 属性
                o3d_geom.vertex_colors = o3d.utility.Vector3dVector(colors_array)
            else:
                raise NotImplementedError(f"Unsupported geometry type: {type(o3d_geom)}")

        # 1. 纯色模式（Open3D原生paint_uniform_color方法）
        if color_mode == "uniform_color":
            o3d_geom.paint_uniform_color(self.uniform_color)

        # 2. 标签颜色模式（data.y为标签，映射到Open3D原生colors属性）
        elif color_mode == "label_color":
            if not hasattr(data, "y"):
                raise AttributeError("color_mode='label_color'时，Data对象必须包含y属性（标签）")
            y_np = data.y.detach().cpu().numpy().squeeze()
            # 预设标签颜色映射（使用self.palette）
            if self.palette is None:
                raise ValueError("color_mode='label_color'时，必须提供palette参数")
            label_color_map = self.palette  # palette应为列表
            # 检查y_np中的标签值是否超出palette范围
            max_label = int(y_np.max())
            if max_label >= len(label_color_map):
                raise ValueError(f"标签值 {max_label} 超出palette范围（最大支持 {len(label_color_map) - 1}）")

            # 单标签（整对象统一颜色）
            if y_np.ndim == 0:
                color = label_color_map[int(y_np)]
                o3d_geom.paint_uniform_color(color)
            # 顶点级标签（逐顶点上色，Open3D原生colors属性）
            else:
                colors = np.array(label_color_map)[y_np.astype(int)]
                set_colors(o3d_geom, colors)

        # 3. 边颜色模式（仅mesh生效，Open3D原生edge_colors属性）
        elif color_mode == "edge_color":
            y_np = data.gt_edge_mask.detach().cpu().numpy().squeeze()
            # 预设标签颜色映射（可扩展）
            label_color_map = {0: (0.5, 0.5, 0.5), 1: (0.0, 1.0, 0.0)}
            # 单标签（整对象统一颜色）
            if y_np.ndim == 0:
                color = label_color_map.get(int(y_np), self.uniform_color)
                o3d_geom.paint_uniform_color(color)
            # 顶点级标签（逐顶点上色，Open3D原生colors属性）
            else:
                colors = [label_color_map.get(int(lab), self.uniform_color) for lab in y_np]
                set_colors(o3d_geom, colors)
        elif color_mode == "tte_color":
            y_np = data.gt_tte_mask.detach().cpu().numpy().squeeze()
            # 预设标签颜色映射（可扩展）
            label_color_map = {0: (0.5, 0.5, 0.5), 1: (0.0, 1.0, 0.0)}
            # 单标签（整对象统一颜色）
            if y_np.ndim == 0:
                color = label_color_map.get(int(y_np), self.uniform_color)
                o3d_geom.paint_uniform_color(color)
            # 顶点级标签（逐顶点上色，Open3D原生colors属性）
            else:
                colors = [label_color_map.get(int(lab), self.uniform_color) for lab in y_np]
                set_colors(o3d_geom, colors)

    def _parse_camera_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解析相机参数为Open3D原生np.float64格式（预设视角/自定义视角）"""

        # 3维向量转换为Open3D原生支持的格式
        def _to_3d_np(v: Vec3) -> np.ndarray:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v_np = np.array(v, dtype=np.float64).squeeze()
            if v_np.shape != (3,):
                raise ValueError(f"相机参数必须是3维向量，当前形状：{v_np.shape}")
            return v_np

        # 预设视角（Open3D原生参数）
        if self.view_mode != "custom":
            cam = self.preset_camera[self.view_mode]
            return _to_3d_np(cam["lookat"]), _to_3d_np(cam["front"]), _to_3d_np(cam["up"])
        # 自定义视角
        else:
            return _to_3d_np(self.lookat), _to_3d_np(self.front), _to_3d_np(self.up)

    def set_view_ctrl(self, vis, geometries):

        # 根据main geom自动聚焦几何对象中心并计算aabb
        # main_geom = geometries['main']
        # aabb = main_geom.get_axis_aligned_bounding_box()
        aabb = None

        def union_aabb(aabb1, aabb2):
            min_bound = np.minimum(aabb1.get_min_bound(), aabb2.get_min_bound())
            max_bound = np.maximum(aabb1.get_max_bound(), aabb2.get_max_bound())
            return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # 遍历 geometries 中的所有几何对象
        for name, geom in geometries.items():
            single_aabb = geom.get_axis_aligned_bounding_box()
            if aabb is None:
                aabb = single_aabb
            else:
                # 合并当前包围盒与联合包围盒
                aabb = union_aabb(aabb, single_aabb)

        # 计算相机距离
        extent = aabb.get_extent()  # 获取包围盒各轴长度
        max_extent = np.max(extent)  # 最长轴距
        distance = max_extent * self.zoom  # 相机距离设为最长轴距的1.5倍

        # 设置相机位置：沿front方向移动distance距离
        ctr = vis.get_view_control()  # 获取Open3D原生ViewControl对象
        _, front, up = self._parse_camera_params()
        lookat = aabb.get_center()
        ctr.set_lookat(lookat)  # Open3D原生方法
        ctr.set_front(front)  # Open3D原生方法
        ctr.set_up(up)  # Open3D原生方法
        ctr.set_zoom(distance / max_extent)  # 调整缩放比例以适应视图
        camera_position = lookat - front * distance

        # 步骤5：打印相机信息
        if self.print_info:
            # 打印包围框信息
            print(f"📏 包围框中心: {aabb.get_center()}")
            print(f"📏 包围框范围: {aabb.get_extent()}")

            # 打印相机信息
            print(f"📷 相机位置: {camera_position}")
            print(f"📷 相机朝向: {front}")
            print(f"📷 相机上方方向: {up}")
            print(f"📷 相机焦点: {lookat}")

    def set_render_opt(self, vis):
        render_opt = vis.get_render_option()  # 获取Open3D原生RenderOption对象
        render_opt.point_size = self.point_size
        render_opt.line_width = self.line_width
        render_opt.background_color = self.background_color
        render_opt.show_coordinate_frame = self.show_coordinate_frame
        render_opt.point_show_normal = self.point_show_normal
        render_opt.mesh_show_wireframe = self.mesh_show_wireframe
        render_opt.mesh_show_back_face = self.mesh_show_back_face
        render_opt.light_on = self.light_on
        #  配置Open3D原生MeshColorOption和MeshShadeOption
        render_opt.mesh_color_option = (
            self.mesh_color_option
            if isinstance(self.mesh_color_option, MeshColorOption)
            else MeshColorOption(self.mesh_color_option)
        )
        render_opt.mesh_shade_option = (
            self.mesh_shade_option
            if isinstance(self.mesh_shade_option, MeshShadeOption)
            else MeshShadeOption(self.mesh_shade_option)
        )

    def capture_img(self, vis, filename):
        if self.save_file:
            # 生成保存路径：root/classname/filename
            save_dir = os.path.join(self.root, self.classname)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename + ".png")
            # Open3D原生保存方法
            vis.capture_screen_image(save_path, do_render=True)
            print(f"✅ 渲染结果已保存（Open3D原生）：{save_path}")
        # 步骤6：显示窗口（Open3D原生run方法）
        if self.window_visible:
            print("📺 显示Open3D原生渲染窗口，关闭窗口后继续...")
            vis.run()


    def get_geometries(self,data):
        """根据data生成main geometry"""
        geometries = {}

        # 步骤1：转换为主几何对象（点云/网格/体素）
        main_geom = self.get_main_geometry(data)
        self.set_geometry_color(main_geom, data, self.color_mode)
        geometries["main"] = main_geom  # 主几何对象

        return geometries
    def forward(self, data):
        # 步骤1：参数校验（含Open3D原生参数范围）
        self.validate_data(data)
        # 步骤2：初始化Open3D原生可视化器，配置窗口参数
        vis = o3d.visualization.Visualizer()
        vis.create_window( width=self.window_size[0], height=self.window_size[1],visible=self.window_visible)

        #-----------------------设置几何对象---------------------------#
        # 步骤2：获取data的几何对象
        geometries = self.get_geometries(data)
        # 根据main geom自动聚焦几何对象中心并计算aabb
        for name, geom in geometries.items():
            vis.add_geometry(geom)
        #------------------------------------------------------------#

        # 步骤4，计算AABB的最长轴距，并设置视口及渲染参数
        # aabb = self.get_scene_aabb(geometries)
        self.set_view_ctrl(vis, geometries)
        # 设置渲染参数
        self.set_render_opt(vis)

        # 更新渲染（Open3D原生流程）
        for name, geom in geometries.items():
            vis.update_geometry(geom)

        vis.update_renderer()

        # 步骤5：保存文件（Open3D原生capture_screen_image方法）
        self.capture_img(vis, data.filename)
        # 步骤6：释放Open3D原生资源
        vis.destroy_window()
        del vis
        # 返回原Data对象（符合BaseTransform接口规范）
        return data



    def __call__(self, data: Data) -> Data:
        return self.forward(data)

class InstCentRenderSaver(VisRenderSaver):
    def __init__(
            self,
            # 新增参数
            cent_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # 中心点颜色
            cent_radius: float = 0.01,  # 中心点半径
            **kwargs
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 初始化新增参数
        self.cent_color = cent_color
        self.cent_radius = cent_radius

    def get_instcent_geometry(self, data):
        """
        创建实例中心点的几何对象（球体）
        :param data: 包含实例中心点坐标的Data对象
        :return: 实例中心点的几何对象列表
        """
        instcent_geometries = {}

        # 检查是否存在实例中心点坐标
        if hasattr(data, 'instance_centroids') and data.instance_centroids is not None and \
            hasattr(data, 'instance_labels') and data.instance_labels is not None:
            # 遍历所有实例中心点坐标
            for coord,label in zip(data.instance_centroids,data.instance_labels):
                if label == 0:
                    continue
                # 创建球体表示实例中心点
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.cent_radius)
                sphere.translate(coord.numpy())  # 将球体移动到中心点位置
                sphere.paint_uniform_color(self.cent_color)  # 设置球体颜色
                sphere.compute_vertex_normals()  # 计算法线以获得更好的渲染效果
                instcent_geometries[label] = sphere

        return instcent_geometries

    def validate_data(self, data: Data):
        """校验Data对象是否包含必要属性"""
        super().validate_data(data)
        if not hasattr(data, 'instance_centroids'):
            raise ValueError("数据缺少instance_centroids属性")

    def get_geometries(self,data):
        """获取所有Open3D原生几何对象，以字典形式返回"""
        geometries = super().get_geometries(data)

        main_geom = geometries['main']
        aabb = main_geom.get_axis_aligned_bounding_box()
        center = aabb.get_center()

        # 步骤2：如果存在实例中心点，则添加到字典中
        inst_cent_geometries = self.get_instcent_geometry(data)
        if inst_cent_geometries is not None:
            # geometries["inst_cent"] = inst_cent_geometries  # 实例中心点
            for label, geom in inst_cent_geometries.items():
                geometries[f"inst_cent_{label}"] = geom

        return geometries

    def __call__(self, data):
        # 调用父类的 __call__ 方法
        return super().__call__(data)


def project_point_to_line(P0, P1, P2):
    """
    将点 P0 投影到由 P1 和 P2 确定的直线上。
    参数:
        P0: 待投影的点 (numpy array, shape=(3,))
        P1: 直线上的第一个点 (numpy array, shape=(3,))
        P2: 直线上的第二个点 (numpy array, shape=(3,))
    返回:
        Proj: 投影点 (numpy array, shape=(3,))
    """
    # 计算方向向量 v = P2 - P1
    v = P2 - P1
    # 计算向量 w = P0 - P1
    w = P0 - P1

    # 计算投影参数 t
    t = np.dot(w, w)

    # 计算投影点
    Proj = P1 - t * v

    return Proj

class RayBlendRenderSaver(InstCentRenderSaver):
    def __init__(
            self,
            proj_cent_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # 中心点颜色
            proj_cent_radius: float = 0.01,  # 中心点半径
            radius_threshold: float = 0.02,  # 半径阈值
            **kwargs
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        self.proj_cent_color = proj_cent_color
        self.proj_cent_radius = proj_cent_radius
        self.radius_threshold = radius_threshold

    def project_inst_cent_to_pointcloud(self, inst_cent_geometries, pointcloud_points):
        """
        将 inst_cent 投影到原始点云数据上，仅在 XY 平面进行搜索。
        """
        projected_geometries = {}

        # 构建 KDTree 用于快速邻域搜索（仅使用 XY 平面）
        pcd_xy = o3d.geometry.PointCloud()
        pointcloud_points_xy = pointcloud_points.copy()
        pointcloud_points_xy[:, 2] = 0  # 将 Z 坐标设为 0
        pcd_xy.points = o3d.utility.Vector3dVector(pointcloud_points_xy)
        kdtree = o3d.geometry.KDTreeFlann(pcd_xy)
        radius = self.radius_threshold  # 半径阈值

        for label, geom in inst_cent_geometries.items():
            # 获取当前 inst_cent 的中心坐标，并将其 Z 坐标设为 0
            inst_cent_coord = np.array(geom.get_center())
            inst_cent_coord_xy = inst_cent_coord.copy()
            inst_cent_coord_xy[2] = 0

            # 使用 radius search 查找 XY 平面上的邻域点
            [k, idx, dists] = kdtree.search_radius_vector_3d(inst_cent_coord_xy, radius)

            if k > 0:
                # 提取邻域点坐标及其距离
                neighbor_points = pointcloud_points[idx, :]
                distances_sq = np.array(dists)  # 距离平方

                # 筛选 Z 值大于等于 inst_cent 的点
                valid_mask = neighbor_points[:, 2] >= inst_cent_coord[2]
                valid_candidates = neighbor_points[valid_mask]
                valid_distances_sq = distances_sq[valid_mask]

                if len(valid_candidates) > 0:
                    # 选取距离最近的点（距离平方最小）
                    closest_idx = np.argmin(valid_distances_sq)
                    projected_point = valid_candidates[closest_idx]
                else:
                    # 没有满足条件的点，保留原始 inst_cent 点
                    projected_point = inst_cent_coord
            else:
                # 没有邻域点，保留原始 inst_cent 点
                print(f"Label {label} has no valid candidates, using original inst_cent.")
                projected_point = inst_cent_coord

            # 创建新的球体表示投影后的点
            projected_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.proj_cent_radius)
            projected_sphere.translate(projected_point)
            projected_sphere.paint_uniform_color(self.proj_cent_color)
            projected_sphere.compute_vertex_normals()

            # 存储投影后的几何对象
            projected_geometries[label] = projected_sphere

        return projected_geometries

    def sort_and_connect_points(self, projected_inst_cents):
        """
        对 projected_inst_cents 进行逆时针排序，并用线段连接相邻点。
        :param projected_inst_cents: dict，键为 label，值为对应的投影点坐标 (x, y, z)
        :return: o3d.geometry.LineSet，表示连接后的线段
        """
        # 提取所有点的坐标
        points = np.array([sphere.get_center() for sphere in projected_inst_cents.values()])  # shape: (N, 3)
        # 计算中心点（质心）
        centroid = np.mean(points, axis=0)
        centroid[1] += 1.0

        # 计算每个点相对于中心点的极角（逆时针）
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

        # 按极角逆时针排序
        sorted_indices = np.argsort(angles)[::-1]  # 逆时针排序
        sorted_points = points[sorted_indices]

        # 构造线段连接相邻点（闭合多边形）
        lines = []
        num_points = len(sorted_points)
        for i in range(0,num_points-1):
            lines.append([i, (i + 1) % num_points])  # 连接当前点与下一个点

        # 创建 LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(sorted_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.0, 0.0, 1.0])  # 红色线段

        return line_set

    def reproj_lineset(self, line_set):
        """
        重新投影 line_set 中的点坐标，并返回新的 LineSet
        :param line_set: o3d.geometry.LineSet 对象
        :return: 新的 o3d.geometry.LineSet 对象
        """
        # 提取点和线段信息
        points = np.asarray(line_set.points)
        lines = np.asarray(line_set.lines)

        # 计算总长度和累积长度
        total_length = 0.0
        cumulative_lengths = [0.0]
        for start_idx, end_idx in lines:
            segment_length = np.linalg.norm(points[end_idx] - points[start_idx])
            total_length += segment_length
            cumulative_lengths.append(total_length)

        # 重新投影点坐标
        new_points = []
        for i, (x, y, z) in enumerate(points):
            new_x = cumulative_lengths[i] - total_length / 2  # 重新计算 x 坐标
            new_y = 0                                        # 固定 y 坐标为 0
            new_z = z                                        # 保留原始 z 坐标
            new_points.append([new_x, new_y, new_z])

        # 创建新的 LineSet
        new_line_set = o3d.geometry.LineSet()
        new_line_set.points = o3d.utility.Vector3dVector(new_points)
        new_line_set.lines = line_set.lines  # 复用原始线段连接关系
        new_line_set.paint_uniform_color([1.0, 1.0, 0.0])  # 设置为红色

        return new_line_set

    def find_nearest_points(self, points, pointcloud_points):
        """
        查找 points 中每个点在 pointcloud_points 中的最近邻点索引。

        :param points: np.ndarray, 形状为 (N, 3)，待查询点集
        :param pointcloud_points: np.ndarray, 形状为 (M, 3)，点云数据
        :return: list[int], 每个点的最近邻点索引
        """
        # 构建 KDTree
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        nearest_indices = []

        # 遍历每个点，查找最近邻点
        for point in points:
            # 查询最近的 1 个点
            [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
            if k > 0:
                nearest_indices.append(idx[0])  # 最近邻点索引
            else:
                nearest_indices.append(None)  # 若未找到有效邻居，则标记为 None

        return nearest_indices


    def get_geometries(self,data):
        """获取所有Open3D原生几何对象，以字典形式返回"""
        geometries = super().get_geometries(data)

        main_geom = geometries['main']
        aabb = main_geom.get_axis_aligned_bounding_box()
        center = aabb.get_center()

        # 获取原始点云数据
        pointcloud_points = data.pos.detach().cpu().numpy().astype(np.float64)

        # 步骤2：如果存在实例中心点，则添加到字典中
        inst_cent_geometries = self.get_instcent_geometry(data)

        if inst_cent_geometries is  None:
            return geometries

        # 质心点
        for label, geom in inst_cent_geometries.items():
            geometries[f"inst_cent_{label}"] = geom

        # 投影实例中心点到点云
        projected_inst_cents = self.project_inst_cent_to_pointcloud(inst_cent_geometries, pointcloud_points)
        for label, geom in projected_inst_cents.items():
            geometries[f"projected_inst_cent_{label}"] = geom

        # 排序并连接的投影质心线段
        line_set = self.sort_and_connect_points(projected_inst_cents)
        geometries["connected_lines"] = line_set
        # # 重新投影排序后的质心线段
        # new_line_set = self.reproj_lineset(line_set)
        # geometries["reproj_connected_lines"] = new_line_set

        if isinstance(main_geom, o3d.geometry.PointCloud):
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(main_geom, alpha=0.1)
        elif isinstance(main_geom, o3d.geometry.TriangleMesh):
            mesh = main_geom
        else:
            raise TypeError(f"Unsupported geometry type for alpha shape: {type(main_geom)}")

        if not mesh.is_edge_manifold() or not mesh.is_vertex_manifold():
            print("警告：网格存在非流形结构，建议修复后再进行变形")
            # 可选：尝试修复网格
            mesh.remove_non_manifold_edges()
            mesh.remove_duplicated_vertices()

        # 查找投影质心在点云中的位置索引
        nearest_idxs = self.find_nearest_points(line_set.points, mesh.vertices)

        # # 使用arap构建新的点云
        # # 转换约束顶点索引和目标位置为 Open3D 支持的类型
        constraint_vertex_indices = o3d.utility.IntVector(nearest_idxs)
        # constraint_vertex_positions = o3d.utility.Vector3dVector(new_line_set.points)
        constraint_vertex_positions = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[nearest_idxs])

        # for pos in constraint_vertex_positions:
        #     pos[1] = pos[1]/20

        def update_projected_positions(constraint_vertex_positions_np, projected_positions):
            """
            根据 constraint_vertex_positions_np 中最小 y 值的点 p，
            计算累积线段长度 L 和到点 p 的累积长度 Lp，
            然后更新 projected_positions 的 x 坐标为 L - Lp。
            参数:
                constraint_vertex_positions_np: np.ndarray, 形状为 (N, 3)，原始点集
                projected_positions: np.ndarray, 形状为 (N, 3)，待更新的投影点集
            返回:
                updated_projected_positions: np.ndarray, 更新后的投影点集
            """
            # 1. 找到最小 y 值的点 p 及其索引
            y_coords = constraint_vertex_positions_np[:, 1]
            min_y_index = np.argmin(y_coords)
            p = constraint_vertex_positions_np[min_y_index]

            # 2. 计算累积线段长度 L
            num_points = len(constraint_vertex_positions_np)
            cumulative_lengths = [0.0]  # 初始累积长度为 0
            total_length = 0.0

            for i in range(num_points - 1):
                segment_length = np.linalg.norm(
                    constraint_vertex_positions_np[i + 1][:2] - constraint_vertex_positions_np[i][:2]
                )
                total_length += segment_length
                cumulative_lengths.append(total_length)

            # 3. 计算到点 p 的累积长度 Lp
            Lp = cumulative_lengths[min_y_index]
            xp = p[0]
            yp = p[1]
            # 4. 更新 projected_positions 的 x 坐标
            updated_projected_positions = projected_positions.copy()
            for i in range(num_points):
                new_x = cumulative_lengths[i] - Lp + xp # 总长度减去到点 p 的累积长度
                updated_projected_positions[i][0] = new_x  # 更新 x 坐标
                updated_projected_positions[i][1] = yp  # 更新 z 坐标

            # updated_projected_positions = 0.5*updated_projected_positions + 0.5*constraint_vertex_positions_np

            return updated_projected_positions

        # 假设 constraint_vertex_positions 已定义（o3d.utility.Vector3dVector 类型）
        constraint_vertex_positions_np = np.asarray(constraint_vertex_positions)  # 转换为 NumPy 数组
        # # 1. 寻找最小 y 值 y0
        # y_coords = constraint_vertex_positions_np[:, 1]  # 提取所有点的 y 坐标
        # y0 = np.min(y_coords)  # 找到最小的 y 值
        # # 2. 形成投影向量 (x, y0, z)
        projected_positions = constraint_vertex_positions_np.copy()  # 复制原始数据
        # projected_positions[:, 1] = y0  # 将所有点的 y 坐标替换为 y0
        # # 如果需要将结果转回 Open3D 格式
        # constraint_vertex_positions = o3d.utility.Vector3dVector(projected_positions)
        updated_positions = update_projected_positions(constraint_vertex_positions_np, projected_positions)
        constraint_vertex_positions = o3d.utility.Vector3dVector(updated_positions)


        # 执行 ARAP 形变

        deformed_geom = mesh.deform_as_rigid_as_possible(
            constraint_vertex_indices=constraint_vertex_indices,
            constraint_vertex_positions=constraint_vertex_positions,
            max_iter=50,
            energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Spokes,
            smoothed_alpha=0.01
        )

        # deformed_geom.paint_uniform_color([0.3, 0.3, 0.3])  # 形变后颜色（红色）
        self.set_geometry_color(deformed_geom, data, "label_color")
        geometries["deformed_geom"] = deformed_geom


        del geometries['main']

        return geometries

    def __call__(self, data):
        # 调用父类的 __call__ 方法
        return super().__call__(data)


class SpeeCurveRenderSaver(InstCentRenderSaver):
    def __init__(
            self,
            **kwargs
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)

    def sort_and_connect_points(self, projected_inst_cents):
        """
        对 projected_inst_cents 进行逆时针排序，并用线段连接相邻点。
        :param projected_inst_cents: dict，键为 label，值为对应的投影点坐标 (x, y, z)
        :return: o3d.geometry.LineSet，表示连接后的线段
        """
        # 提取所有点的坐标
        points = np.array([sphere.get_center() for sphere in projected_inst_cents.values()])  # shape: (N, 3)
        # 计算中心点（质心）
        centroid = np.mean(points, axis=0)
        centroid[1] += 1.0

        # 计算每个点相对于中心点的极角（逆时针）
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

        # 按极角逆时针排序
        sorted_indices = np.argsort(angles)[::-1]  # 逆时针排序
        sorted_points = points[sorted_indices]

        # 构造线段连接相邻点（闭合多边形）
        lines = []
        num_points = len(sorted_points)
        for i in range(0,num_points-1):
            lines.append([i, (i + 1) % num_points])  # 连接当前点与下一个点

        # 创建 LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(sorted_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.0, 0.0, 1.0])  # 红色线段

        return line_set

    def get_geometries(self,data):
        """获取所有Open3D原生几何对象，以字典形式返回"""
        geometries = super().get_geometries(data)

        main_geom = geometries['main']
        aabb = main_geom.get_axis_aligned_bounding_box()
        center = aabb.get_center()

        # 获取原始点云数据
        pointcloud_points = data.pos.detach().cpu().numpy().astype(np.float64)

        # 步骤2：如果存在实例中心点，则添加到字典中
        inst_cent_geometries = self.get_instcent_geometry(data)

        if inst_cent_geometries is  None:
            return geometries

        # 质心点
        for label, geom in inst_cent_geometries.items():
            geometries[f"inst_cent_{label}"] = geom

        # 排序并连接的投影质心线段
        line_set = self.sort_and_connect_points(inst_cent_geometries)
        geometries["connected_lines"] = line_set

        if isinstance(main_geom, o3d.geometry.PointCloud):
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(main_geom, alpha=0.1)
        elif isinstance(main_geom, o3d.geometry.TriangleMesh):
            mesh = main_geom
        else:
            raise TypeError(f"Unsupported geometry type for alpha shape: {type(main_geom)}")

        def encode_mesh_to_smile_curve(mesh_vertices, smile_curve_points):
            """
            Encoder阶段：计算每个mesh顶点到smile_curve上最近点的信息。
            使用向量化操作提升效率。

            参数:
                mesh_vertices: np.ndarray, 形状为 (N, 3)，mesh顶点坐标
                smile_curve_points: np.ndarray, 形状为 (M, 3)，smile_curve上的点坐标

            返回:
                nearest_indices: np.ndarray, 形状为 (N,)，每个mesh顶点对应的smile_curve最近点索引
                offsets: np.ndarray, 形状为 (N, 3)，每个顶点到最近点的偏移向量
            """
            # 计算所有mesh顶点到所有smile_curve点的距离矩阵
            # shape: (N, M)
            distances = np.linalg.norm(mesh_vertices[:, np.newaxis, :] - smile_curve_points[np.newaxis, :, :], axis=2)

            # 找到每个mesh顶点最近的smile_curve点索引
            nearest_indices = np.argmin(distances, axis=1)

            # 获取最近点坐标
            nearest_points = smile_curve_points[nearest_indices]

            # 计算偏移向量
            offsets = nearest_points - mesh_vertices

            return nearest_indices, offsets


        def deform_smile_curve(constraint_vertex_positions_np):
            y_coords = constraint_vertex_positions_np[:, 1]
            min_y_index = np.argmin(y_coords)
            p = constraint_vertex_positions_np[min_y_index]

            # 2. 计算累积线段长度 L
            num_points = len(constraint_vertex_positions_np)
            cumulative_lengths = [0.0]  # 初始累积长度为 0
            total_length = 0.0

            for i in range(num_points - 1):
                segment_length = np.linalg.norm(
                    constraint_vertex_positions_np[i + 1][:2] - constraint_vertex_positions_np[i][:2]
                )
                total_length += segment_length
                cumulative_lengths.append(total_length)

            # 3. 计算到点 p 的累积长度 Lp
            Lp = cumulative_lengths[min_y_index]
            xp = p[0]
            yp = p[1]
            # 4. 更新 projected_positions 的 x 坐标
            updated_projected_positions = constraint_vertex_positions_np.copy()
            for i in range(num_points):
                new_x = cumulative_lengths[i] - Lp + xp  # 总长度减去到点 p 的累积长度
                updated_projected_positions[i][0] = new_x  # 更新 x 坐标
                updated_projected_positions[i][1] = yp  # 更新 z 坐标

            # updated_projected_positions = 0.5*updated_projected_positions + 0.5*constraint_vertex_positions_np

            return updated_projected_positions

        def decode_mesh_from_deformed_curve(nearest_indices, offsets, deformed_curve_points):
            """
            Decoder阶段：根据形变后的smile_curve和偏移向量，重新计算mesh顶点坐标。

            参数:
                nearest_indices: list[int], 每个mesh顶点对应的smile_curve最近点索引
                offsets: np.ndarray, 形状为 (N, 3)，每个顶点到最近点的偏移向量
                deformed_curve_points: np.ndarray, 形状为 (M, 3)，形变后的smile_curve点坐标

            返回:
                deformed_mesh_vertices: np.ndarray, 形状为 (N, 3)，形变后的mesh顶点坐标
            """
            deformed_mesh_vertices = []
            for i, index in enumerate(nearest_indices):
                # 获取形变后smile_curve上的最近点
                deformed_nearest_point = deformed_curve_points[index]
                # 根据偏移向量恢复顶点位置
                deformed_vertex = deformed_nearest_point - offsets[i]
                deformed_mesh_vertices.append(deformed_vertex)

            return np.array(deformed_mesh_vertices)

        def interpolate_points_on_lineset(line_set, num_interpolated_points=3):
            """
            在 LineSet 的每条线段上均匀内插指定数量的点。
            参数:
                line_set: o3d.geometry.LineSet，输入的线段集合
                num_interpolated_points: int，每条线段上需要内插的点数
            返回:
                new_points: np.ndarray，更新后的点集（包含原始点和内插点）
                new_lines: np.ndarray，更新后的线段连接关系
            """
            # 提取原始点和线段信息
            original_points = np.asarray(line_set.points)
            lines = np.asarray(line_set.lines)

            new_points = []
            new_lines = []

            point_index_offset = len(original_points)  # 新点的起始索引

            # 遍历每条线段
            for start_idx, end_idx in lines:
                start_point = original_points[start_idx]
                end_point = original_points[end_idx]

                # 生成内插点
                interpolated_points = []
                for i in range(1, num_interpolated_points + 1):
                    t = i / (num_interpolated_points + 1)  # 插值参数
                    interp_point = (1 - t) * start_point + t * end_point
                    interpolated_points.append(interp_point)

                # 添加原始起点和终点
                segment_points = [start_point] + interpolated_points + [end_point]
                new_points.extend(segment_points)

                # 更新线段连接关系
                segment_indices = list(range(len(new_points) - len(segment_points), len(new_points)))
                for i in range(len(segment_indices) - 1):
                    new_lines.append([segment_indices[i], segment_indices[i + 1]])

            # 转换为 Open3D 支持的数据结构
            new_points = np.array(new_points)
            new_lines = np.array(new_lines)

            return new_points, new_lines


        mesh_vertices = np.asarray(mesh.vertices)
        smile_curve_points, _ = interpolate_points_on_lineset(line_set, num_interpolated_points=10)
        # smile_curve_points = np.asarray(line_set.points)
        # 1. Encoder阶段：编码mesh顶点信息
        nearest_indices, offsets = encode_mesh_to_smile_curve(mesh_vertices, smile_curve_points)

        # 2. 形变阶段：对smile_curve进行形变
        deformed_curve_points = deform_smile_curve(smile_curve_points)

        # 3. Decoder阶段：解码mesh顶点新位置
        deformed_mesh_vertices = decode_mesh_from_deformed_curve(nearest_indices, offsets, deformed_curve_points)

        mesh.vertices = o3d.utility.Vector3dVector(deformed_mesh_vertices)

        self.set_geometry_color(mesh, data, "label_color")
        geometries["deformed_geom"] = mesh


        del geometries['main']

        return geometries


    def __call__(self, data):
        # 调用父类的 __call__ 方法
        return super().__call__(data)


def main():
    # 1. 构造测试Data对象（包含pos/face/filename/y/edge_index）
    pos = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=torch.float32)  # (8,3)顶点
    face = torch.tensor([
        [0, 1, 2], [0, 2, 3], [1, 5, 6], [1, 6, 2],
        [5, 4, 7], [5, 7, 6], [4, 0, 3], [4, 3, 7],
        [3, 2, 6], [3, 6, 7], [4, 5, 1], [4, 1, 0]
    ], dtype=torch.int32).T  # (3,12)面片

    # labels
    y = torch.randint(low=0, high=3, size=(8,))
    # edge_labels
    gt_edge_labels = torch.randint(low=0, high=2, size=(8,))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int32)  # 边索引
    data = Data(
        pos=pos,
        face=face,
        edge_index=edge_index,
        y=y,  # 标签（对应蓝色）
        gt_edge_labels=gt_edge_labels, # 边标签（对应红色）
        filename="cube_mesh_labelcolor.png"  # 保存文件名
    )
    # 2. 初始化RenderAndSaver（使用Open3D原生参数）
    renderer = VisRenderSaver(
        save_file=True,  # 保存渲染结果
        root="./render_demo",
        classname="cube",
        data_mode="mesh",  # 网格模式
        color_mode="edge_color",  # 标签颜色模式
        uniform_color=(0.2, 0.6, 0.9),  # 默认纯色
        window_visible=False,  # 后台渲染（不弹窗口）
        window_size=(1000, 800),
        # 视角配置（Open3D原生）
        view_mode="custom",
        lookat=[0.0, 0.0, 0.0],
        front=[1.0, 1.0, 1.0],
        up=[0.0, 1.0, 0.0],
        # Open3D原生RenderOption参数
        point_size=5.0,
        mesh_show_wireframe=False,  # Open3D原生布尔值
        mesh_color_option=1,  # Open3D原生值：3-Vertex（纯顶点色）
        mesh_shade_option=0,  # Open3D原生值：1-Smooth（高斯平滑）
        light_on=False,
    )
    # 3. 调用（符合BaseTransform接口，传入Data对象）
    data = renderer(data)
    # 4. 扩展测试：显示窗口+纯色模式（Open3D原生参数）
    renderer2 = VisRenderSaver(
        save_file=True,
        root="./render_demo",
        classname="cube",
        data_mode="point",  # 点云模式
        color_mode="uniform_color",
        window_visible=False,  # 显示Open3D原生窗口
        view_mode="front",
        background_color=(0.1, 0.1, 0.1),
        point_show_normal=True,
        mesh_color_option=0,  # Open3D原生默认值（仅mesh生效）
        mesh_shade_option=0,  # Open3D原生默认值（仅mesh生效）
    )
    data.filename = "cube_point_uniformcolor"  # 替换文件名
    data = renderer2(data)


# ===================== 测试示例（使用Open3D原生参数）=====================
if __name__ == "__main__":
    main()