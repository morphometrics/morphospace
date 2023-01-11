import networkx as nx
import numpy as np
import raster_geometry as rg
from prediction_to_skeleton_graph import (
    compute_geometric_graph,
    merge_graph_at_points,
    prediction_to_skeleton_graph,
    reduce_graph,
    threshold_image,
)


def test_threshold_image():
    """Test thresholding image with known in- and ouput"""
    # Setup
    input_image = np.array([[0, 1, 0], [0.2, 0.8, 0.03], [0.1, 1, 1]])
    output_image = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])
    # Excercise
    result_image = threshold_image(input_image, 0.3) * 1

    # Verify
    assert np.array_equal(result_image, output_image)


def test_compute_geometric_graph():
    """Test if test image is converted to the right geometric graph"""
    # Setup
    input_image = np.array(
        [[False, True, False], [False, True, False], [False, True, True]]
    )
    out_graph = nx.Graph()
    out_graph.add_edge(1, 2)
    out_graph.add_edge(2, 3)
    out_graph.add_edge(3, 4)
    nx.set_node_attributes(
        out_graph,
        {
            1: np.array([0, 1]),
            2: np.array([1, 1]),
            3: np.array([2, 1]),
            4: np.array([2, 2]),
        },
        "pos",
    )

    out_graph_pos = nx.get_node_attributes(out_graph, "pos")

    # Excercise
    result_graph = compute_geometric_graph(input_image, radius=1)
    result_graph_pos = nx.get_node_attributes(result_graph, "pos")

    # remove positions as the functions cant compare arrays as node attributes
    nx.set_node_attributes(out_graph, None, "pos")
    nx.set_node_attributes(result_graph, None, "pos")

    # test positions
    assert np.array_equal(
        list(out_graph_pos.values()), list(result_graph_pos.values())
    )
    # test graph structure
    assert nx.utils.graphs_equal(result_graph, out_graph)


def test_merge_graph_at_points():
    # Setup
    # create a test graph
    cylinder_image = rg.cylinder(15, 13, 2)
    branch_point_coordinates_test = np.array([[7.0, 7.0, 1.0]])
    end_point_coordinates_test = np.array([[7.0, 7.0, 13]])

    graph = compute_geometric_graph(cylinder_image)
    graph_pos = nx.get_node_attributes(graph, "pos")
    point_index = [
        node
        for node, position in graph_pos.items()
        if np.array_equal(np.array([position]), branch_point_coordinates_test)
        or np.array_equal(np.array([position]), end_point_coordinates_test)
    ]

    # Excercise
    graph_merged = merge_graph_at_points(
        graph,
        cylinder_image,
        branch_point_coordinates=branch_point_coordinates_test,
        end_point_coordinates=end_point_coordinates_test,
    )
    # graph_merged_pos = nx.get_node_attributes(graph_merged, 'pos')

    # too uncreative to think of a simple way to chekc for correct merging...
    # This will do, though
    # Test if nodes merged
    assert graph.number_of_nodes() > graph_merged.number_of_nodes()
    # Test if neighbour relationship of merged nodes changed
    assert len(list(graph.edges(point_index))) < len(
        list(graph_merged.edges(point_index))
    )


def test_reduce_graph():

    # Setup
    # generate toy graph
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
        ]
    )
    node_positions = {
        1: np.array([0, 0, 0]),
        2: np.array([1, 1, 1]),
        3: np.array([2, 2, 2]),
        4: np.array([3, 3, 3]),
        5: np.array([4, 4, 4]),
        6: np.array([5, 5, 5]),
        7: np.array([6, 6, 6]),
        8: np.array([7, 7, 7]),
        9: np.array([8, 8, 8]),
        10: np.array([9, 9, 9]),
    }
    nx.set_node_attributes(graph, node_positions, name="pos")

    # Reduce to 2 branch points and two end points
    branch_point_coordinates = np.array([[3, 3, 3], [6, 6, 6]])
    end_point_coordinates = np.array([[0, 0, 0], [9, 9, 9]])
    graph_reduced = reduce_graph(
        graph, branch_point_coordinates, end_point_coordinates
    )

    edge_attributes = nx.get_edge_attributes(
        graph_reduced, "points_inbetween"
    ).values()
    edge_attributes = np.array(list(edge_attributes)).reshape(6, 3)
    removed_nodes = np.array(
        [v for n, v in node_positions.items() if n not in graph_reduced.nodes]
    )

    # check if the number of edges has been remove to the expected number
    assert graph_reduced.number_of_edges() == 3
    # check if the remaining nodes are branch and end points
    assert all(
        np.allclose(graph_reduced.nodes[n]["pos"], v)
        for n, v in node_positions.items()
        if n in graph_reduced.nodes
    )
    # check if edge attributes contain all removed pixel positions
    assert (
        np.sort(removed_nodes, axis=0) == np.sort(edge_attributes, axis=0)
    ).all()


def test_prediction_to_skeleton_graph():
    # create cylinder image and branch and enpoint at its ends
    cylinder_image = rg.cylinder(15, 13, 2)
    branch_point_coordinates_test = np.array([[7.0, 7.0, 1.0]])
    end_point_coordinates_test = np.array([[7.0, 7.0, 13.0]])
    tested_skeleton_graph = prediction_to_skeleton_graph(
        cylinder_image,
        branch_point_coordinates_test,
        end_point_coordinates_test,
    )
    # just test if it runs and gives expected graph structure.
    # Rest is tested before
    assert tested_skeleton_graph.number_of_edges() == 1
    assert tested_skeleton_graph.number_of_nodes() == 2
