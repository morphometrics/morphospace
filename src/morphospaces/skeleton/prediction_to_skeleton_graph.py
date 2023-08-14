import igl
import napari_process_points_and_surfaces as npps
import networkx as nx
import numpy as np
import trimesh as tri
from scipy.spatial import KDTree


def threshold_image(
    skeleton_prediction_image: np.array, threshold: float = 0.3
) -> np.array:
    """Threshold image to only use only use pixels below the threshold
        probability

    Parameters
    ----------
    skeleton_prediction_image : ndarray(z,y,x)
        image of the probability space for the skeleton
        predicted by the neural network.
    threshold : float
        Threshold is the fraction of the maximum value of the image
        Range 0 (no pixels) to 1(all pixels)

    Returns
    -------
    ndarray(z,y,x)
         Masked input image
    """
    thresh = np.max(skeleton_prediction_image) * threshold
    skeleton_mask = np.zeros_like(skeleton_prediction_image, dtype=bool)
    skeleton_mask[skeleton_prediction_image >= thresh] = True
    return skeleton_mask


def compute_geometric_graph(
    skeleton_prediction_image: np.array, radius: int = 1
) -> nx.Graph:
    """Takes an image and creates edges to all pixels in a radius to generate a
        geometric graph.
        Pixel coordinates are stored in node attribute "pos".

    Parameters
    ----------
    skeleton_prediction_image : ndimage(z,x,y)
        binary image to compute the graph from.
    radius : int, optional
        radius to connect pixel with neighbours , by default 1

    Returns
    -------
    nx.graph
        fully connected pixel graph
    """

    # get pixel coordinates from binary image
    coordinates = np.array(np.argwhere(skeleton_prediction_image)).astype(int)
    indices = [i for i in range(1, len(coordinates) + 1)]
    # use coordinates as positions
    pos = {}
    for i, n in enumerate(indices):
        pos[n] = coordinates[i]
    # generate geometric graph
    graph = nx.random_geometric_graph(n=indices, radius=radius, dim=3, pos=pos)
    # remove isolated pixels
    graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def merge_graph_at_points(
    graph: nx.Graph,
    skeleton_prediction_image: np.array,
    branch_point_coordinates: np.array,
    end_point_coordinates: np.array,
) -> nx.Graph:
    """Merge graph nodes close to branch/edge points (or any given list of
        points, really).
        Computes a surface mesh of the input image, computes the distance of
        each input point
        to the image and merges all the nodes in that radius.
        Resulting Graph will lead through all branch and end points.

    Parameters
    ----------
    graph : nx.Graph
            Graph with attribute 'pos' holding the pixel positon in an array.

    skeleton_prediction_image : ndarray(z,y,x)
        binary label of the probability space for the  skeleton
        predicted by the neural network

    branch_point_coordinates : ndarray(n,3)
         Array containing the point coordinates of n branch points
    end_point_coordinates : ndarray(n,3)
         Array containing the point coordinates of n end points

    Returns
    -------
    nx.Graph:
        Pixel graph leading through all input points

    """

    # drop dependencies on npps and igl
    mesh = npps.label_to_surface(skeleton_prediction_image)
    mesh = tri.Trimesh(mesh[0], mesh[1])

    signed_distance_bp, _, _ = igl.signed_distance(
        branch_point_coordinates, mesh.vertices, mesh.faces
    )
    signed_distance_ep, _, _ = igl.signed_distance(
        end_point_coordinates, mesh.vertices, mesh.faces
    )

    # merge points around branch and end points to gurantee graph passing
    # through them
    # identifying points to merge

    # node identifiers
    branch_points = []
    end_points = []

    # node coordinates
    branch_point_coordinates_s = []
    end_point_coordinates_s = []

    for node, positions in graph.nodes(data="pos"):
        for bp in branch_point_coordinates:
            if np.allclose(positions.astype(int), bp.astype(int)):
                branch_points.append(node)
                branch_point_coordinates_s.append(positions)
        for ep in end_point_coordinates:
            if np.allclose(positions.astype(int), ep.astype(int)):
                end_points.append(node)
                end_point_coordinates_s.append(positions)

    branch_point_coordinates = np.array(branch_point_coordinates_s)
    end_point_coordinates = np.array(end_point_coordinates_s)

    point_indices_joined = branch_points + end_points

    point_coordinates_joined = np.concatenate(
        (branch_point_coordinates, end_point_coordinates)
    )

    point_distances_joined = np.concatenate(
        [np.atleast_1d(signed_distance_bp), np.atleast_1d(signed_distance_ep)]
    )

    nodes_to_merge = []

    node_pos = nx.get_node_attributes(graph, "pos")
    kd_tree_nodes = KDTree(list(node_pos.values()))

    for i, point in enumerate(point_coordinates_joined):

        nodes_to_merge.append(
            kd_tree_nodes.query_ball_point(point, r=point_distances_joined[i])
        )

    print(len(nodes_to_merge))
    # merge points
    graph_merged = graph.copy()

    for point_index, node_to_merge in zip(
        point_indices_joined, nodes_to_merge
    ):
        for j in node_to_merge:
            nx.contracted_nodes(
                graph_merged, point_index, j + 1, copy=False, self_loops=False
            )

    for point_index, i in enumerate(point_indices_joined):
        nx.set_node_attributes(
            graph_merged,
            {i: point_coordinates_joined[point_index]},
            name="pos",
        )

    return graph_merged


def reduce_graph(
    graph: nx.Graph,
    branch_point_coordinates: np.array,
    end_point_coordinates: np.array,
) -> nx.Graph:
    """
    Reduce the graph to branch- and end points. Pixels lying inbetween
    are stored as edge attribute in an ndarray(n_pixels,3).

    Parameters
    ----------
    graph : nx.Graph
        pixel graph to be reduced
    branch_point_coordinates : ndarray(n,3)
         Array containing the point coordinates of n branch points
    end_point_coordinates : ndarray(n,3)
         Array containing the point coordinates of n end points


    Returns
    -------
    nx.Graph
        Reduces input graph
    """

    branch_points = []
    end_points = []

    branch_point_coordinates_s = []
    end_point_coordinates_s = []

    for node, positions in graph.nodes(data="pos"):
        for bp in branch_point_coordinates:
            if np.allclose(positions.astype(int), bp.astype(int)):
                branch_points.append(node)
                branch_point_coordinates_s.append(positions)
        for ep in end_point_coordinates:
            if np.allclose(positions.astype(int), ep.astype(int)):
                end_points.append(node)
                end_point_coordinates_s.append(positions)

    point_indices_joined = branch_points + end_points

    # find shortest path in image from branchpoints to endpoints
    sps = []
    for bp in branch_points:
        for point in point_indices_joined:
            if point == bp:
                continue
            else:
                sps.append(nx.shortest_path(graph, bp, point))

    # remove all connections including all points.
    # Very slow, maybe do in one step?
    for num_nodes_in_path in sps:
        if len(np.intersect1d(num_nodes_in_path, point_indices_joined)) > 2:
            sps.remove(num_nodes_in_path)

    nodes_to_store = {}
    node_coordinates = nx.get_node_attributes(graph, "pos")
    for edge in sps:
        node_to_store = [
            node for node in edge if node not in [edge[0], edge[-1]]
        ]
        node_to_store = np.array([node_coordinates[n] for n in node_to_store])
        nodes_to_store[(edge[0], edge[-1])] = node_to_store

    # create the reduced graph
    graph_reduced = nx.Graph()

    for n1, n2 in nodes_to_store:
        graph_reduced.add_edge(n1, n2)
    nx.set_edge_attributes(graph_reduced, nodes_to_store, "points_inbetween")
    reduced_positions = {}
    for node in graph_reduced.nodes:
        reduced_positions[node] = node_coordinates[node]
    nx.set_node_attributes(graph_reduced, reduced_positions, name="pos")

    return graph_reduced


def prediction_to_skeleton_graph(
    skeleton_prediction_image: np.array,
    branch_point_coordinates: np.array,
    end_point_coordinates: np.array,
) -> nx.Graph:
    """
    Takes the predicted skeleton, branch points and end points from the
    neural network and computes a graph connecting the points and stores
    all pixels position inbetween as edge attributes.

    Parameters
    ----------
    skeleton_prediction_image : ndarray(z,y,x)
        image of the probability space for the skeleton
        predicted by the neural network.
    branch_point_coordinates : ndarray(n,3)
         Array containing the point coordinates of n branch points
    end_point_coordinates : ndarray(n,3)
         Array containing the point coordinates of n end points

    Returns
    -------
    nx.Graph
        Reduces pixel graph containing all branch and endpoints
    """

    skeleton_prediction_image_thresh = threshold_image(
        skeleton_prediction_image, threshold=0.3
    )
    geometric_graph = compute_geometric_graph(skeleton_prediction_image_thresh)
    merged_graph = merge_graph_at_points(
        geometric_graph,
        skeleton_prediction_image_thresh,
        branch_point_coordinates,
        end_point_coordinates,
    )
    reduced_graph = reduce_graph(
        merged_graph, branch_point_coordinates, end_point_coordinates
    )
    return reduced_graph
