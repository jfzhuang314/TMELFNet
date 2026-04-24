import numpy as np
import open3d as o3d
from scipy import sparse as sp


class MeshProc:
    @staticmethod
    def get_boundary_points(mesh):
        """
        该函数用于从输入的 mesh 中提取边界点的索引
        :param mesh: 一个 Open3D 的 TriangleMesh 对象
        :return: 边界点的索引数组
        """
        # 获取三角形面的索引
        triangles = np.asarray(mesh.triangles)

        # 生成所有边的列表
        edges = []
        for triangle in triangles:
            edges.append([triangle[0], triangle[1]])
            edges.append([triangle[1], triangle[2]])
            edges.append([triangle[2], triangle[0]])
        edges = np.array(edges)

        # 对边进行排序
        sorted_edges = np.sort(edges, axis=1)

        # 找出唯一的边及其出现次数
        unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)

        # 找出边界边
        boundary_edges = unique_edges[counts == 1]

        # 提取边界点的索引
        boundary_points_indices = np.unique(boundary_edges.flatten())

        return boundary_points_indices

    @staticmethod
    def get_adjacency_matrix(mesh):
        """
        该函数用于计算并返回 mesh 的邻接矩阵
        :param mesh: 一个 Open3D 的 TriangleMesh 对象
        :return: 邻接矩阵，类型为 scipy.sparse.csr_matrix
        """
        # 获取顶点数量
        num_vertices = len(mesh.vertices)
        # 获取三角形面的索引
        triangles = np.asarray(mesh.triangles)

        # 存储非零元素的行索引、列索引和值
        row_indices = []
        col_indices = []
        values = []

        # 遍历每个三角形面
        for triangle in triangles:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        # 记录非零元素的行索引、列索引和值
                        row_indices.append(triangle[i])
                        col_indices.append(triangle[j])
                        values.append(1)

        # 创建稀疏矩阵
        adjacency_matrix = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

        return adjacency_matrix

    @staticmethod
    def get_average_edge_length(mesh):
        """
        此函数用于计算Open3D Mesh对象的平均边长。

        :param mesh: 一个Open3D的TriangleMesh对象
        :return: Mesh的平均边长
        """
        # 初始化边长总和和边的数量
        total_edge_length = 0
        edge_count = 0

        # 获取顶点和三角形面的数组
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 遍历每个三角形面
        for triangle in triangles:
            # 获取三角形的三个顶点
            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]

            # 计算三条边的长度
            edge_lengths = [
                np.linalg.norm(v0 - v1),
                np.linalg.norm(v1 - v2),
                np.linalg.norm(v2 - v0)
            ]

            # 累加边长总和
            total_edge_length += sum(edge_lengths)
            # 累加边的数量
            edge_count += 3

        # 计算平均边长
        if edge_count > 0:
            average_edge_length = total_edge_length / edge_count
        else:
            average_edge_length = 0

        return average_edge_length

    @staticmethod
    def get_multiple_khop_neighbors(query_vidxs, adjacency_matrix, k):
        """
        该函数用于获取指定顶点的 k 阶邻接点
        :param query_vidxs: 待查询的顶点索引
        :param k: 阶数
        :param adjacency_matrix: 邻接矩阵，类型为 scipy.sparse.csr_matrix
        :return: 包含 k 阶邻接点索引的集合
        """
        all_k_hop_neighbors = set()
        for vertex_index in query_vidxs:
            k_hop_neighbors = MeshProc.get_single_khop_neighbors(vertex_index, k, adjacency_matrix)
            all_k_hop_neighbors.update(k_hop_neighbors)
        return all_k_hop_neighbors
    @staticmethod
    def get_single_khop_neighbors(vertex_index, k, adjacency_matrix):
        """
        该函数用于获取指定顶点的 k 阶邻接点
        :param vertex_index: 指定顶点的索引
        :param k: 阶数
        :param adjacency_matrix: 邻接矩阵，类型为 scipy.sparse.csr_matrix
        :return: 包含 k 阶邻接点索引的集合
        """
        neighbors = set([vertex_index])
        current_neighbors = set([vertex_index])
        for _ in range(k):
            new_neighbors = set()
            for neighbor in current_neighbors:
                # 使用稀疏矩阵的 nonzero() 方法获取非零元素的索引
                row_neighbors = adjacency_matrix[neighbor].nonzero()[1]
                new_neighbors.update(row_neighbors)
            neighbors.update(new_neighbors)
            current_neighbors = new_neighbors
        neighbors.remove(vertex_index)
        return neighbors

    @staticmethod
    def remove_vertices(mesh, vertices_to_remove):
        """
        Remove the specified vertices from the mesh, delete the related faces, and update the triangle indices.
        :param mesh: Input Open3D TriangleMesh object
        :param vertices_to_remove: List of vertex indices to be removed
        :return: Processed mesh
        """
        # Convert vertices_to_remove to a list of integers
        vertices_to_remove = list(map(int, vertices_to_remove))

        # Mark the vertices to be retained
        num_vertices = len(mesh.vertices)
        vertex_mask = np.ones(num_vertices, dtype=bool)
        vertex_mask[vertices_to_remove] = False

        # Get the retained vertices
        new_vertices = np.asarray(mesh.vertices)[vertex_mask]

        # Filter out the faces that contain the removed vertices
        triangles = np.asarray(mesh.triangles)
        valid_triangles = []
        for triangle in triangles:
            if all(vertex_mask[triangle]):
                valid_triangles.append(triangle)
        valid_triangles = np.array(valid_triangles)

        # Remap vertex indices
        new_vertex_indices = np.cumsum(vertex_mask) - 1
        for i in range(len(valid_triangles)):
            valid_triangles[i] = new_vertex_indices[valid_triangles[i]]

        # Update the vertices and faces of the mesh
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        new_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)

        # If the mesh has a 'vLabel' attribute, set the values at the positions of k_hop_point_cloud to 0
        if hasattr(mesh, 'vLabel'):
            v_label = np.array(mesh.vLabel)
            v_label[list(vertices_to_remove)] = 0
            new_mesh.vLabel = v_label

        return new_mesh
