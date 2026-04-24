import numpy as np
from scipy.interpolate import interpolate


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


def mean_scalar_normalize(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    zmax = np.max(points[:, 2])
    points[:, 2] = points[:, 2] -zmax
    points /= 60.0
    return points

def bounding_box(points):
    """
    :param points: [n,3]
    :return: [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    return np.array([points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
                     points[:, 0].max(), points[:, 1].max(), points[:, 2].max()])

def bounding_box_lenght(points):
    return np.array([points[:, 0].max() - points[:, 0].min(),
                     points[:, 1].max() - points[:, 1].min(),
                     points[:, 2].max() - points[:, 2].min()])


def max_abs_axis_val(points, axis=0):
    """
    :param points: [n,3],要求为中心化后的点
    :param axis: 0: x轴, 1: y轴, 2: z轴, 默认0轴
    :return: max_abs_axis_val
    """
    return np.maximum(points[:, axis].max(), np.abs(points[:, axis].min()))

def perform_hist_eq(x, y, options=None):
    """
    执行直方图均衡化

    参数:
        x: 输入数组
        y: 目标分布数组或字符串 ('gaussian' 或 'linear')
        options: 配置字典，支持以下选项:
            - cols: 沿列操作 (bool)
            - rows: 沿行操作 (bool)
            - dim3: 沿第三维操作 (bool)
            - absval: 仅对绝对值操作 (bool)
            - histinterp: 直方图插值系数 [0,1] (float)

    返回:
        x: 均衡化后的数组
    """
    if options is None:
        options = {}

    # 处理列表/元组输入
    if isinstance(x, (list, tuple)):
        return [perform_hist_eq(xi, yi, options) for xi, yi in zip(x, y)]

    # 获取选项
    absval = options.get('absval', False)
    cols = options.get('cols', False)
    rows = options.get('rows', False)
    dim3 = options.get('dim3', False)
    lambda_val = options.get('histinterp', 1.0)

    if cols and rows:
        raise ValueError('不能同时指定 cols 和 rows')

    x = np.array(x, dtype=np.float64)
    original_shape = x.shape

    # 沿列操作
    if cols and x.shape[1] > 1:
        if isinstance(y, str):
            y_target = y
        else:
            y = np.array(y, dtype=np.float64)
            if x.shape[1] != y.shape[1]:
                raise ValueError('x 和 y 必须有相同的列数')

        result = np.zeros_like(x)
        for i in range(x.shape[1]):
            if isinstance(y, str):
                result[:, i] = perform_hist_eq(x[:, i], y, options)
            else:
                result[:, i] = perform_hist_eq(x[:, i], y[:, i], options)
        return result

    # 沿行操作
    if rows and x.shape[0] > 1:
        if isinstance(y, str):
            y_target = y
        else:
            y = np.array(y, dtype=np.float64)
            if x.shape[0] != y.shape[0]:
                raise ValueError('x 和 y 必须有相同的行数')

        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            if isinstance(y, str):
                result[i, :] = perform_hist_eq(x[i, :], y, options)
            else:
                result[i, :] = perform_hist_eq(x[i, :], y[i, :], options)
        return result

    # 沿第三维操作
    if dim3 and len(x.shape) >= 3 and x.shape[2] > 1:
        if isinstance(y, str):
            y_target = y
        else:
            y = np.array(y, dtype=np.float64)
            if x.shape[2] != y.shape[2]:
                raise ValueError('x 和 y 必须有相同的第三维大小')

        result = np.zeros_like(x)
        for i in range(x.shape[2]):
            if isinstance(y, str):
                result[:, :, i] = perform_hist_eq(x[:, :, i], y, options)
            else:
                result[:, :, i] = perform_hist_eq(x[:, :, i], y[:, :, i], options)
        return result

    # 展平数组
    x_flat = x.flatten()

    # 处理字符串目标分布
    if isinstance(y, str):
        if y == 'gaussian':
            y_flat = np.random.randn(len(x_flat))
        elif y == 'linear':
            y_flat = np.linspace(0, 1, len(x_flat))
        else:
            raise ValueError(f'未知的密度类型: {y}')
    else:
        y = np.array(y, dtype=np.float64)
        y_flat = y.flatten()

    # 处理绝对值模式
    if absval:
        s = np.sign(x_flat)
        x_flat = np.abs(x_flat)
        y_flat = np.abs(y_flat)

    # 排序
    Ix = np.argsort(x_flat)
    Iy = np.argsort(y_flat)

    vx = x_flat[Ix]
    vy = y_flat[Iy]

    nx = len(x_flat)
    ny = len(y_flat)

    # 插值
    ax = np.linspace(1, ny, nx)
    ay = np.arange(1, ny + 1)

    # 使用线性插值
    interp_func = interpolate.interp1d(ay, vy, kind='linear', fill_value='extrapolate')
    vx_interp = interp_func(ax)

    # 混合原始值和插值结果
    x_sorted = np.zeros_like(x_flat)
    x_sorted[Ix] = (1 - lambda_val) * x_flat[Ix] + lambda_val * vx_interp

    # 恢复符号
    if absval:
        x_sorted = x_sorted * s

    # 恢复原始形状
    x_result = x_sorted.reshape(original_shape)

    return x_result
