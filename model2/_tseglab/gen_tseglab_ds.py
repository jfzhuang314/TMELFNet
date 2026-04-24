import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from PIL.Image import Image
from omegaconf import DictConfig
from pympler import asizeof

from configs.hydra_yaml_config import get_yaml_config, print_yaml_cfg_dict, instantiate_from_config
from datasets import T3DSPlusSegDataset, TADPMSegDataset
from datasets.T3DSPlusSegDataset import T3DSPlusSegReader
from external_libs import gen_utils as gu
import open3d as o3d

import pymeshlab as ml
import os
from curvature import compute_curvature
from utils.points_utils import compute_knn_average_distance


# def visualize_curvature(source_path: str, target_path: str):
#     ms = ml.MeshSet()
#
#     ms.load_new_mesh(source_path)
#     ms.load_filter_script('Compute_curvature_principal_directions.mlx')
#     ms.apply_filter_script()
#
#     ms.save_current_mesh(target_path, binary=False)
#

def create_mesh_from_pygdata(data_obj, color=[0.5, 0.5, 0.5]):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data_obj.pos.cpu().numpy().astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(data_obj.face.T.cpu().numpy().astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.compute_adjacency_list()
    mesh.paint_uniform_color(color)
    return mesh

def off_screen_render(mesh, bg_color=[1.0, 1.0, 1.0], save_path="intensity.png"):

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=1080,
        height=640,
        visible=False
    )

    vis.add_geometry(mesh)

    # 渲染选项配置
    render_option = vis.get_render_option()
    render_option.light_on = True  # 关闭灯光
    render_option.mesh_show_wireframe = False  # 不显示线框
    # render_option.mesh_shade_option = o3d.visualization.rendering.MeshShadeOption.kDefault
    render_option.background_color = np.array(bg_color)  # 设置背景颜色

    # 调整视角以适合模型
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 2])  # 设置视角方向
    ctr.set_lookat([0, 0, 0])  # 设置观察中心
    ctr.set_up([0, 1, 0])  # 设置向上方向
    ctr.set_zoom(0.5)  # 设置缩放


    vis.poll_events()
    vis.update_renderer()
    # 直接拿像素
    image = vis.capture_screen_float_buffer(False)
    pixels = np.asarray(image)

# 可选：保存截图
    if save_path:
        vis.capture_screen_image(save_path, do_render=True)
        print(f"截图已保存到：{save_path}")


    vis.destroy_window()



    return pixels


def off_screen_render_pure(mesh, bg_color=[1.0, 1.0, 1.0], save_path="pure.png"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=1080,
        height=640,
        visible=False
    )

    vis.add_geometry(mesh)

    # 渲染选项配置
    render_option = vis.get_render_option()
    render_option.light_on = False  # 关闭灯光
    render_option.mesh_show_wireframe = False  # 不显示线框
    # render_option.mesh_shade_option = o3d.visualization.rendering.MeshShadeOption.kDefault
    render_option.background_color = np.array(bg_color)  # 设置背景颜色

    # 调整视角以适合模型
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 2])  # 设置视角方向
    ctr.set_lookat([0, 0, 0])  # 设置观察中心
    ctr.set_up([0, 1, 0])  # 设置向上方向
    ctr.set_zoom(0.5)  # 设置缩放

    vis.poll_events()
    vis.update_renderer()
    # 直接拿像素
    image = vis.capture_screen_float_buffer(False)
    pixels = np.asarray(image)

    # 可选：保存截图
    if save_path:
        vis.capture_screen_image(save_path, do_render=True)
        print(f"截图已保存到：{save_path}")


    vis.destroy_window()

    return pixels


def visualize_mesh_without_wireframe_and_light(mesh, window_name="3D Visualization", bg_color=[1.0, 1.0, 1.0],
                                               save_path=None):
    """
    使用 Open3D 可视化网格，不显示线框并去除灯光效果

    Args:
        mesh: Open3D TriangleMesh 对象
        window_name: 窗口标题
        bg_color: 背景颜色 RGB，默认白色 [1.0, 1.0, 1.0]
        save_path: 可选，保存截图的路径

    Returns:
        None
    """
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_name,
        width=1080,
        height=640,
        visible=False
    )

    vis.add_geometry(mesh)

    # 渲染选项配置
    render_option = vis.get_render_option()
    render_option.light_on = False  # 关闭灯光
    render_option.mesh_show_wireframe = False  # 不显示线框
    # render_option.mesh_shade_option = o3d.visualization.rendering.MeshShadeOption.kDefault
    render_option.background_color = np.array(bg_color)  # 设置背景颜色
    render_option.point_size = 1.0  # 点大小
    render_option.show_coordinate_frame = False  # 不显示坐标轴

    # 调整视角以适合模型
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 2])  # 设置视角方向
    ctr.set_lookat([0, 0, 0])  # 设置观察中心
    ctr.set_up([0, 1, 0])  # 设置向上方向
    ctr.set_zoom(0.5)  # 设置缩放

    # 可选：保存截图
    if save_path:
        vis.capture_screen_image(save_path, do_render=True)
        print(f"截图已保存到：{save_path}")

    # 运行可视化
    # vis.run()
    vis.destroy_window()


def curvature_to_color(curvature, method='sigmoid'):
    # 归一化曲率到 [0,1] 用于颜色映射

    norm_curv = normalizate_field(curvature, method)

    # 使用 jet 配色
    colors = np.zeros((len(norm_curv), 3))
    colors[:, 0] = norm_curv  # R
    colors[:, 1] = norm_curv  # G
    colors[:, 2] = norm_curv  # B
    return colors


def normalizate_field(curvature, method):
    if method == 'percentile':
        # 使用 2% 和 98% 百分位数，避免极端值影响
        curv_min = np.percentile(curvature, 5)
        curv_max = np.percentile(curvature, 95)
        print(f"curv_min: {curv_min}, curv_max: {curv_max}")

        norm_curv = (curvature - curv_min) / (curv_max - curv_min + 1e-8)
        # 截断超出范围的值
        norm_curv = np.clip(norm_curv, 0, 1)

    elif method == 'sigmoid':
        # 使用 sigmoid 归一化，适合有正负的曲率
        k = 1.0 / (np.std(curvature) + 1e-8)
        norm_curv = 1 / (1 + np.exp(-k * (curvature - np.median(curvature))))

    else:  # minmax
        # 原始方法（不推荐）
        curv_min = np.min(curvature)
        curv_max = np.max(curvature)
        norm_curv = (curvature - curv_min) / (curv_max - curv_min + 1e-8)
    return norm_curv


def test_T3DSPlusSegReader():
    print("Current working directory:", os.getcwd())

    # config_path = "./configs"
    # config_name = "test_T3DSPlusDataset"
    # cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    # args = cfg.paths
    # ds_args = cfg.dataset

    config_path = "../../../configs"
    config_name = "test_TSeglab"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    path_args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    # print_yaml_config(config_path=config_path, config_name=config_name)

    reader = T3DSPlusSegReader(is_filter_small_instance=ds_args.is_filter_small_instance,
                               filter_small_instance_size=ds_args.filter_small_instance_size,
                               verbose=True)

    # 01KTRG9K_lower 0OF8OOCX_lower 1QI2XWMD_lower IKZNMT4Z_upper 01K17AN8_lower 4G9LHQ2X_lower
    # 特别案例 01K3G866_lower B5GFZIRW_lower FHA6N618_lower
    # 01M10NMV_lower 4J24X0ES_upper AVI68NCT_upper B5GFZIRW_lower CD81F1F4_upper CQZX40I5_upper
    # E552BK9X_upper  GNR6QR3P_lower TVSR5QBQ_lower U9UBDO05_upper VB7GYS2O_upper Y48DURWV_upper ZD89X7G1_lower

    name = "01FG729R_lower"
    LU = name.split('_')[1]
    patentt_id = name.split('_')[0]
    obj_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.obj"
    json_name = f"{LU}/{patentt_id}/{patentt_id}_{LU}.json"

    obj_filename = os.path.join(path_args.source_obj_data_path, obj_name)
    seg_json_filename = os.path.join(path_args.source_seg_json_data_path, json_name)


    data = reader(obj_filename, seg_json_filename)
    print(data.keys())

    print(cfg.pre_transform_obj)


    pre_transform = instantiate_from_config(cfg.pre_transform_obj)
    transform = instantiate_from_config(cfg.transform_obj)
    pre_filter = instantiate_from_config(cfg.pre_filter_obj)

    print(pre_transform, transform, pre_filter)

    data = pre_transform(data)
    print(data.keys())

    step_foward(data, path_args)


def step_foward(data, path_args, split="train"):
    mesh = create_mesh_from_pygdata(data, color=[0.5, 0.5, 0.5])

    tseglab_tmp_path = path_args.tseglab_tmp_path.replace("__split__", split)
    tseglab_images_path = path_args.tseglab_images_path.replace("__split__", split)
    tseglab_masks_path = path_args.tseglab_masks_path.replace("__split__", split)

    # 检查文件是否已存在
    image_exists = os.path.exists(os.path.join(tseglab_images_path, data.filename + '.png'))
    mask_exists = os.path.exists(os.path.join(tseglab_masks_path, data.filename + '.png'))

    if image_exists and mask_exists:
        print(f"⏭️  跳过已存在的文件：{data.filename}")
        return

    print(f"\n🔄 开始处理：{data.filename} (split={split})")

    # 计算强度图像素
    intensity_path = os.path.join(tseglab_tmp_path, data.filename + '_intensity.png')
    os.makedirs(os.path.dirname(intensity_path), exist_ok=True)
    intensity_pixels = off_screen_render(mesh, save_path=intensity_path)

    # 计算曲率图像素
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    knn_avg_dist = compute_knn_average_distance(torch.from_numpy(tm.vertices), 3).numpy()
    radius = 5 * knn_avg_dist
    from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure
    gauss = discrete_gaussian_curvature_measure(tm, np.asarray(tm.vertices), radius)
    mean = discrete_mean_curvature_measure(tm, np.asarray(tm.vertices), radius)
    # vertex_colors = curvature_to_color(gauss)
    curv = normalizate_field(gauss, 'percentile')
    vertex_colors = np.vstack([curv, curv, curv]).T
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    curv_path = os.path.join(tseglab_tmp_path, data.filename + '_curv.png')
    os.makedirs(os.path.dirname(curv_path), exist_ok=True)
    curv_pixels = off_screen_render(mesh, save_path=curv_path)

    # 计算深度图像素
    depth = normalizate_field(np.asarray(tm.vertices)[:, 2], 'percentile')
    vertex_colors = np.vstack([depth, depth, depth]).T
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    depth_path = os.path.join(tseglab_tmp_path, data.filename + '_depth.png')
    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
    depth_pixels = off_screen_render(mesh, save_path=depth_path)
    Ir = intensity_pixels
    Ir[:, :, 1] = curv_pixels[:, :, 0]
    Ir[:, :, 2] = depth_pixels[:, :, 0]
    img = (np.array(Ir) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    Ir_path = os.path.join(tseglab_images_path, data.filename + '.png')
    os.makedirs(os.path.dirname(Ir_path), exist_ok=True)
    cv2.imwrite(Ir_path, img)

    # gt mesh可视化
    gt_labels = data.y.squeeze().cpu().numpy()
    gt_mesh = gu.get_colored_mesh(mesh, gt_labels.astype(np.int32), 'label15')
    gt_path = os.path.join(tseglab_tmp_path, data.filename + '_gt.png')
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    visualize_mesh_without_wireframe_and_light(gt_mesh, save_path=gt_path)
    generate_gt_mask(gt_path, tseglab_tmp_path, tseglab_masks_path, data.filename)


def generate_gt_mask(gt_path, tmp_path, mask_path, file_name):

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
        [ 0.3, 0.3, 0.0 ],      # 9
        [ 0.15, 0.3, 0.0 ],     # 10
        [ 0.0, 0.3, 0.3 ],      # 11
        [ 0.0, 0.0, 0.3 ],      # 12
        [ 0.3, 0.0, 0.3 ],      # 13
        [ 0.3, 0.0, 0.15 ],      # 14
        # [ 1.0, 1.0, 1.0]        # 15, 白色背景需要单独处理
    ])

    img_path = gt_path
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    h, w, _ = img.shape
    total_pixels = h * w

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print("📊 步骤 1: 提取唯一颜色...")
    # =====================
    # 步骤 1: 提取唯一颜色及出现次数
    # =====================
    # 重塑为 (h*w, 3)
    img_flat = img.reshape(-1, 3)

    # 使用 np.unique 找出所有唯一颜色（按行去重）
    unique_colors, inverse_indices, counts = np.unique(
        img_flat,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    n_unique = len(unique_colors)
    # print(f"✅ 图像中包含 {n_unique} 种唯一颜色")

    # =====================
    # 过滤：只保留出现次数 > 10 的颜色
    # =====================
    min_count_threshold = 10
    valid_mask = counts >= min_count_threshold

    valid_unique_colors = unique_colors[valid_mask]
    valid_counts = counts[valid_mask]

    # 创建映射：原索引 -> 新索引（只包含有效颜色）
    valid_indices_map = {}
    new_idx = 0
    for old_idx in range(n_unique):
        if valid_mask[old_idx]:
            valid_indices_map[old_idx] = new_idx
            new_idx += 1

    n_valid = len(valid_unique_colors)
    # print(f"✅ 出现次数 > {min_count_threshold} 的颜色有 {n_valid} 种")

    # # 统计信息（只显示有效的）
    # print("\n📋 唯一颜色列表及出现次数（计数 > 10）:")
    # for i, (color, count) in enumerate(zip(valid_unique_colors, valid_counts)):
    #     percentage = count / total_pixels * 100
    #     print(f"  [{i:3d}] RGB: [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}] - "
    #           f"出现 {count:6d} 次 ({percentage:5.2f}%)")

    # print("\n🎯 步骤 2: 匹配调色板颜色...")
    # =====================
    # 步骤 2: 找出每个唯一颜色最接近的调色板索引
    # =====================
    palette_indices = np.zeros(n_valid, dtype=np.uint8)

    for i, color in enumerate(valid_unique_colors):
        # 计算与所有调色板颜色的距离
        distances = np.linalg.norm(label15_palte - color, axis=1)

        # 找到最近的索引
        closest_idx = np.argmin(distances)
        palette_indices[i] = closest_idx

        closest_color = label15_palte[closest_idx]
        distance = distances[closest_idx]

    #     print(f"  颜色 [{i:3d}] -> 调色板索引 {closest_idx:2d} "
    #           f"(距离={distance:.4f})")
    #
    # print("\n🔧 步骤 3: 生成 GT Mask...")
    # =====================
    # 步骤 3: 根据映射关系生成 mask
    # =====================
    # inverse_indices: 每个像素对应的唯一颜色索引 (h*w,)
    # palette_indices: 每个唯一颜色对应的调色板索引 (n_valid,)

    # 创建新的 inverse_indices，将无效颜色映射到背景色（索引 0）
    mask_flat = np.zeros(len(inverse_indices), dtype=np.uint8)

    for old_idx, new_idx in valid_indices_map.items():
        pixel_mask = (inverse_indices == old_idx)
        mask_flat[pixel_mask] = palette_indices[new_idx]

    # 重塑回 (h, w)
    mask = mask_flat.reshape(h, w)

    # print(f"✅ Mask 形状：{mask.shape}")
    # print(f"✅ Mask 值范围：[{mask.min()}, {mask.max()}]")
    # print(f"✅ 唯一标签值：{np.unique(mask)}")

    # 详细统计
    # print("\n📊 GT Mask 类别统计:")
    unique_labels, label_counts = np.unique(mask, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        percentage = count / total_pixels * 100
        color = label15_palte[label]
        # print(f"  类别 {label:2d}: {count:6d} 像素 ({percentage:5.2f}%) - "
        #       f"RGB: [{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]")

    # 保存结果
    print("\n💾 保存结果...")

    # 单通道 mask
    mask_uint8 = mask.astype(np.uint8)
    mask_path = os.path.join(mask_path, file_name + '.png')
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask_uint8)
    # print(f"✅ 单通道 Mask 已保存到：{mask_path}")

    # cv2.imshow("img", mask_uint8*15)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # 彩色 mask（用于可视化）
    mask_colored = label15_palte[mask]
    mask_colored = (mask_colored * 255).astype(np.uint8)

    # cv2.imshow("img", mask_colored)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    mask_colored_path = os.path.join(tmp_path, file_name + '_mask_colored.png')
    os.makedirs(os.path.dirname(mask_colored_path), exist_ok=True)
    cv2.imwrite(mask_colored_path, cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
    # print(f"✅ 彩色 Mask 已保存到：{mask_colored_path}")


def prepare_data(cfg_obj: DictConfig):
    if cfg_obj.dataset.name == "T3DSPlusSegDataset":
        dm, train_loader, test_loader \
            = T3DSPlusSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)
    elif cfg_obj.dataset.name == "TADPMSegDataset":
        dm, train_loader, test_loader \
            = TADPMSegDataset.get_dataloaders_from_cfg_obj(cfg_obj)

    return dm, train_loader, test_loader

def batch_process():
    print("Current working directory:", os.getcwd())

    config_path = "../../configs"
    config_name = "test_TSeglab"
    cfg = get_yaml_config(config_path=config_path, config_name=config_name)
    path_args = cfg.paths
    ds_args = cfg.dataset
    print_yaml_cfg_dict(cfg)

    cfg_obj = instantiate_from_config(cfg)
    print(cfg_obj)

    _, train_loader, test_loader = prepare_data(cfg_obj)

    print(train_loader, test_loader)

    for batch in train_loader:
        data = batch[0]
        step_foward(data, path_args, split="train")

    # for batch in test_loader:
    #     data = batch[0]
    #     # print(data)
    #     step_foward(data, path_args, split="test")


def main():
    # test_T3DSPlusSegReader()
    # generate_gt_mask()
    batch_process()



if __name__ == "__main__":
    main()