import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import csr_matrix


# -----------------------------------------------------------------------------
# 主函数：compute_curvature（Python 版，和你的 MATLAB 完全对应）
# -----------------------------------------------------------------------------
def compute_curvature(V, F, options=None):
    """
    输入：
        V: (3, N) 顶点矩阵
        F: (3, M) 面矩阵（索引从 0 开始！）
        options: dict 可选项
            curvature_smoothing: 平滑迭代次数 default=3
            verb: 是否显示进度 default=0
    输出：
        Umin, Umax: (3, N) 主曲率方向
        Cmin, Cmax: (N,) 主曲率值
        Cmean: (N,) 平均曲率
        Cgauss: (N,) 高斯曲率
        Normal: (3, N) 顶点法向量
    """
    orient = 1
    if options is None:
        options = {}

    naver = options.get('curvature_smoothing', 3)
    verb = options.get('verb', 0)

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=int)
    n = V.shape[1]
    m = F.shape[1]

    # -------------------------------------------------------------------------
    # 构建边-邻接面关系
    # -------------------------------------------------------------------------
    i = np.concatenate([F[0], F[1], F[2]])
    j = np.concatenate([F[1], F[2], F[0]])
    s = np.tile(np.arange(m), 3)
    Af = csr_matrix((s, (i, j)), shape=(n, n))

    # -------------------------------------------------------------------------
    # 提取有向边 E -> (face1, face2)
    # -------------------------------------------------------------------------
    I, J, S1 = sp.sparse.find(Af)
    I2, J2, S2 = sp.sparse.find(Af.T)
    mask = (S1 > 0) & (S2 > 0)
    I = I[mask]
    J = J[mask]
    E = np.stack([S1[mask], S2[mask]], axis=1)
    dir_mask = I < J
    E = E[dir_mask]
    i = I[dir_mask]
    j = J[dir_mask]
    ne = len(i)

    # -------------------------------------------------------------------------
    # 边方向 + 长度
    # -------------------------------------------------------------------------
    e = V[:, j] - V[:, i]
    d = np.linalg.norm(e, axis=0)
    e = e / d[None, :]

    # -------------------------------------------------------------------------
    # 面法向量
    # -------------------------------------------------------------------------
    normal_f = compute_normal(V, F)

    # -------------------------------------------------------------------------
    # 计算二面角（带符号）
    # -------------------------------------------------------------------------
    n1 = normal_f[:, E[:, 0]]
    n2 = normal_f[:, E[:, 1]]
    dp = np.sum(n1 * n2, axis=0)
    dp = np.clip(dp, -1, 1)
    beta = np.arccos(dp)

    cp = np.cross(n1.T, n2.T).T
    si = orient * np.sign(np.sum(cp * e, axis=0))
    beta = beta * si

    # -------------------------------------------------------------------------
    # 曲率张量 T
    # -------------------------------------------------------------------------
    T = np.zeros((3, 3, ne), dtype=np.float64)
    w = d * beta
    for x in range(3):
        for y in range(x + 1):
            T[x, y] = e[x] * e[y] * w
            T[y, x] = T[x, y]

    # -------------------------------------------------------------------------
    # 张量累加到顶点
    # -------------------------------------------------------------------------
    Tv = np.zeros((3, 3, n), dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    for k in range(ne):
        Tv[:, :, i[k]] += T[:, :, k]
        Tv[:, :, j[k]] += T[:, :, k]
        weight[i[k]] += 1
        weight[j[k]] += 1

    # -------------------------------------------------------------------------
    # 顶点面积归一化
    # -------------------------------------------------------------------------
    a = V[:, F[2]] - V[:, F[0]]
    b = V[:, F[1]] - V[:, F[0]]
    ab = np.cross(a.T, b.T).T
    area_f = np.linalg.norm(ab, axis=0) / 2.0

    row = np.tile(np.arange(m), 3)
    col = F.flatten()
    data = np.tile(area_f, 3)
    U_sp = csr_matrix((data, (row, col)), shape=(m, n))
    Av = np.asarray(U_sp.sum(axis=0)).squeeze()
    Av[Av < 1e-8] = 1.0

    for x in range(3):
        for y in range(3):
            Tv[x, y] /= Av[None, None, :]

    # -------------------------------------------------------------------------
    # 曲率张量平滑
    # -------------------------------------------------------------------------
    for _ in range(naver):
        for x in range(3):
            for y in range(3):
                val = Tv[x, y].reshape(-1)
                val = perform_mesh_smoothing(F, V, val)
                Tv[x, y] = val.reshape(1, 1, n)

    # -------------------------------------------------------------------------
    # 特征分解 → 曲率
    # -------------------------------------------------------------------------
    Umin = np.zeros((3, n), dtype=np.float64)
    Umax = np.zeros((3, n), dtype=np.float64)
    Cmin = np.zeros(n, dtype=np.float64)
    Cmax = np.zeros(n, dtype=np.float64)
    Normal = np.zeros((3, n), dtype=np.float64)

    for k in range(n):
        M = Tv[:, :, k]
        try:
            d, u = np.linalg.eig(M)
            d = np.real(d)
            u = np.real(u)
        except:
            Umin[:, k] = [1, 0, 0]
            Umax[:, k] = [0, 1, 0]
            Normal[:, k] = [0, 0, 1]
            continue

        idx = np.argsort(np.abs(d))
        d = d[idx]
        u = u[:, idx]

        Normal[:, k] = u[:, 0]
        Umin[:, k] = u[:, 2]
        Umax[:, k] = u[:, 1]
        Cmin[k] = d[2]
        Cmax[k] = d[1]

    # -------------------------------------------------------------------------
    # 确保 Cmin ≤ Cmax
    # -------------------------------------------------------------------------
    swap = Cmin > Cmax
    Cmin[swap], Cmax[swap] = Cmax[swap], Cmin[swap]
    Umin[:, swap], Umax[:, swap] = Umax[:, swap].copy(), Umin[:, swap].copy()

    # -------------------------------------------------------------------------
    # 法向量方向校正
    # -------------------------------------------------------------------------
    n_vertex = compute_normal(V, F)
    sign_correct = np.sign(np.sum(Normal * n_vertex, axis=0))
    Normal *= sign_correct[None, :]

    Cmean = (Cmin + Cmax) / 2.0
    Cgauss = Cmin * Cmax

    return Umin, Umax, Cmin, Cmax, Cmean, Cgauss, Normal


# -----------------------------------------------------------------------------
# 辅助函数：compute_normal（面法向量）
# -----------------------------------------------------------------------------
def compute_normal(V, F):
    v1 = V[:, F[0]]
    v2 = V[:, F[1]]
    v3 = V[:, F[2]]
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1.T, e2.T).T
    n = n / (np.linalg.norm(n, axis=0) + 1e-8)[None, :]
    return n


# -----------------------------------------------------------------------------
# 辅助函数：perform_mesh_smoothing（简单拉普拉斯平滑）
# -----------------------------------------------------------------------------
def perform_mesh_smoothing(F, V, val):
    n = V.shape[1]
    i = np.concatenate([F[0], F[1], F[2]])
    j = np.concatenate([F[1], F[2], F[0]])
    A = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    degree = np.asarray(A.sum(axis=1)).squeeze()
    degree[degree < 1] = 1.0
    val_smooth = (A @ val) / degree
    return val_smooth