import igl
import napari_accelerated_pixel_and_object_classification as npps
import networkx as nx
import numpy as np
import trimesh as tri


# Split up the big function to make it more modular
def threshold_image(skeleton_prediction_image, threshold):
    """Threshold image to only use only use pixels below the threshold
        probability

    Args:
        skeleton_prediction_image (ndarray(z,y,x)): image of the probability
                                                    space for the skeleton
                                                    predicted by the neural
                                                    network.
        threshold (float): percentage of pixels used.
                            Range 0(no pixels) to 1(all pixels)

    Returns:
        ndimage(z,y,x): Thresholded input image
    """
    thresh = np.max(skeleton_prediction_image) * threshold
    skeleton_prediction_image[skeleton_prediction_image < thresh] = 0
    skeleton_prediction_image[skeleton_prediction_image >= thresh] = 1

    return skeleton_prediction_image


def compute_geometric_graph(skeleton_prediction_image, radius=1):
    """Takes an image and creates edges to all pixels in a radius to generate a
        geometric graph.
        Pixel coordinates are stored in node attribute "pos".
    Args:
        skeleton_prediction_image (ndimage(z,x,y)): binary image to compute the
        graph from.
        radius(int): radius in pixels to connect neighbouring pixels

    Returns:
        nx.graph: fully connected pixel graph
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
    graph,
    skeleton_prediction_image,
    branch_point_coordinates,
    end_point_coordinates,
):
    """Merge graph nodes close to branch/edge points (or any given list of
        points, really).
        Computes a surface mesh of the input image, computes the distance of
        each input point
        to the image and merges all the nodes in that radius.
        Resulting Graph will lead through all branch and end points.

    Args:
        graph (nx.Graph): Graph with attribute 'pos' holding the pixel positon
                            in an array.
        skeleton_prediction_image (ndarray(z,y,x)): binary label of the
                                                    probability space for the
                                                    skeleton predicted by the
                                                    neural network

        branch_point_coordinates (ndarray(n,3)): Array containing the point
                                                    coordinates of
                                                    n branch points
        end_point_coordinates (ndarray(n,3)): Array containing the
                                                        point coordinates
                                                        of n end points

    Returns:
        nx.Graph: Pixel graph leading through all input points
    """
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
    for d, points in zip(point_distances_joined, point_coordinates_joined):
        distances = np.linalg.norm(
            points - np.array(list(dict(graph.nodes(data="pos")).values())),
            axis=1,
        )
        nodes_to_merge.append(np.argwhere(distances <= d).flatten() + 1)
    # merge points
    graph_merged = graph.copy()

    for p, ntm in zip(point_indices_joined, nodes_to_merge):
        for j in ntm:
            nx.contracted_nodes(
                graph_merged, p, j, copy=False, self_loops=False
            )

    for p, i in enumerate(point_indices_joined):
        nx.set_node_attributes(
            graph_merged, {i: point_coordinates_joined[p]}, name="pos"
        )

    return graph_merged


def reduce_graph(graph, branch_point_coordinates, end_point_coordinates):
    """Reduce the graph, so that we have a edge with one pixel.

    Args:
        graph (nx.Graph):pixel graph to be reduced
        branch_point_coordinates (ndarray(n,3)): Array containing the point
                                                coordinates of n branch points
        end_point_coordinates (ndarray(n,3)): Array containing the point
                                                coordinates of n end points

    Returns:
        nx.Graph: reduced pixel graph
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
        for ep in end_points:
            sps.append(nx.shortest_path(graph, bp, ep))

    # remove all connections including all points.
    # Very slow, maybe do in one step?
    for path_length in sps:
        if len(np.intersect1d(path_length, point_indices_joined)) > 2:
            sps.remove(path_length)

    # create the reduced graph
    graph_reduced = nx.Graph()

    for sp in sps:
        for i in range(len(sp) - 1):
            graph_reduced.add_edge(sp[i], sp[i + 1])

    reduced_positions = {}
    for node in graph_reduced.nodes:
        reduced_positions[node] = dict(graph.nodes(data="pos"))[node]
    nx.set_node_attributes(graph_reduced, reduced_positions, name="pos")

    return graph_reduced
