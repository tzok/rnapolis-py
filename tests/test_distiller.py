import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rnapolis.distiller import (
    assign_to_representatives,
    build_knn_graph,
    build_clustering_from_groups,
    build_facility_location_selection_curve,
    build_hierarchical_selection,
    build_output_data,
    build_radius_graph_candidates,
    connected_components_from_adjacency,
    determine_optimal_representative_count,
    get_clustering_at_cutoff,
    parse_arguments,
    pairwise_squared_l2,
    validate_cli_arguments,
)


def test_build_clustering_from_groups_orders_sizes_consistently():
    file_paths = [Path(name) for name in ["a.cif", "b.cif", "c.cif", "d.cif"]]
    distance_matrix = np.array(
        [
            [0.0, 0.1, 5.0, 5.0],
            [0.1, 0.0, 5.0, 5.0],
            [5.0, 5.0, 0.0, 0.2],
            [5.0, 5.0, 0.2, 0.0],
        ]
    )

    clustering = build_clustering_from_groups(
        [[2, 3], [0, 1]], distance_matrix, file_paths
    )

    assert clustering["n_clusters"] == 2
    assert clustering["cluster_sizes"] == [2, 2]
    assert clustering["clusters"] == [
        {"representative": "a.cif", "members": ["b.cif"]},
        {"representative": "c.cif", "members": ["d.cif"]},
    ]


def test_assign_to_representatives_groups_by_nearest_exemplar():
    distance_matrix = np.array(
        [
            [0.0, 0.2, 1.5, 1.8],
            [0.2, 0.0, 1.0, 1.7],
            [1.5, 1.0, 0.0, 0.3],
            [1.8, 1.7, 0.3, 0.0],
        ]
    )

    groups = assign_to_representatives(distance_matrix, [0, 2])

    assert groups == {0: [0, 1], 2: [2, 3]}


def test_facility_location_selection_curve_and_auto_count():
    gains = [10.0, 4.0, 1.0, 0.2, 0.05, 0.01]

    curve = build_facility_location_selection_curve(gains)

    assert curve[0] == {
        "value": 1,
        "n_clusters": 1,
        "marginal_gain": 10.0,
        "remaining_gain": 5.26,
    }
    assert curve[-1]["remaining_gain"] == 0.0

    n_representatives = determine_optimal_representative_count(gains)
    assert 1 <= n_representatives <= len(gains)


def test_get_clustering_at_cutoff_returns_unified_schema():
    file_paths = [Path(name) for name in ["a.cif", "b.cif", "c.cif"]]
    distance_matrix = np.array(
        [
            [0.0, 0.1, 2.0],
            [0.1, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    linkage_matrix = np.array(
        [
            [0.0, 1.0, 0.1, 2.0],
            [2.0, 3.0, 2.0, 3.0],
        ]
    )

    clustering = get_clustering_at_cutoff(
        linkage_matrix, distance_matrix, file_paths, 0.5
    )

    assert clustering == {
        "n_clusters": 2,
        "cluster_sizes": [2, 1],
        "clusters": [
            {"representative": "a.cif", "members": ["b.cif"]},
            {"representative": "c.cif", "members": []},
        ],
    }


def test_build_output_data_uses_unified_top_level_schema():
    output = build_output_data(
        mode="approximate",
        method="facility-location",
        distance_metric="pca-l2",
        n_structures=3,
        clustering={"n_clusters": 1, "cluster_sizes": [3], "clusters": []},
        selection=build_hierarchical_selection(
            value=1.5, unit="pca-l2", source="auto-detected"
        ),
        diagnostics={"selection_curve": []},
        parameters={"pca_dimensions": 2},
    )

    assert output == {
        "parameters": {
            "mode": "approximate",
            "method": "facility-location",
            "distance_metric": "pca-l2",
            "n_structures": 3,
            "pca_dimensions": 2,
        },
        "selection": {
            "variable": "distance_cutoff",
            "value": 1.5,
            "unit": "pca-l2",
            "source": "auto-detected",
            "rule": "exponential-decay-knee",
        },
        "clustering": {"n_clusters": 1, "cluster_sizes": [3], "clusters": []},
        "diagnostics": {"selection_curve": []},
    }


def test_build_knn_graph_is_symmetric_and_reports_stats():
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ]
    )

    adjacency, edges, diagnostics = build_knn_graph(embedding, 2)

    assert adjacency[0] == {1}
    assert adjacency[2] == {3}
    assert len(edges) == 2
    assert diagnostics["n_neighbors"] == 2
    assert diagnostics["n_edges"] == 2
    assert diagnostics["avg_degree"] == 1.0
    assert diagnostics["neighbor_search"] == "sklearn"


def test_build_knn_graph_accepts_explicit_search_engine():
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ]
    )

    _, _, diagnostics = build_knn_graph(embedding, 2, neighbor_search="sklearn")

    assert diagnostics["neighbor_search"] == "sklearn"


def test_connected_components_from_adjacency_splits_graph():
    adjacency = [{1}, {0}, {3}, {2}, set()]

    assert connected_components_from_adjacency(adjacency) == [[0, 1], [2, 3], [4]]


def test_build_radius_graph_candidates_returns_monotone_candidate_values():
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ]
    )
    file_paths = [Path(name) for name in ["a.cif", "b.cif", "c.cif", "d.cif"]]

    candidates, diagnostics = build_radius_graph_candidates(embedding, file_paths, 2)

    values = [entry["value"] for entry in candidates]
    cluster_counts = [entry["n_clusters"] for entry in candidates]
    assert values == sorted(values)
    assert cluster_counts[0] >= cluster_counts[-1]
    assert diagnostics["n_neighbors"] == 2


def test_pairwise_squared_l2_returns_dense_distance_matrix():
    embedding = np.array([[0.0, 0.0], [3.0, 4.0]])

    distance_matrix = pairwise_squared_l2(embedding)

    assert np.allclose(distance_matrix, np.array([[0.0, 5.0], [5.0, 0.0]]))


def test_parse_arguments_accepts_visualize_output_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["distiller", "--visualize", "results.png", "example.cif"],
    )

    args = parse_arguments()

    assert args.visualize == "results.png"
    assert args.files == [Path("example.cif")]


def test_parse_arguments_requires_visualize_output_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["distiller", "--visualize"])

    with pytest.raises(SystemExit):
        parse_arguments()


def test_validate_cli_arguments_rejects_graph_backend_with_dense_only_methods():
    args = SimpleNamespace(
        n_neighbors=32,
        neighbor_search="sklearn",
        n_representatives=None,
        method="hierarchical",
        mode="approximate",
        threshold=None,
        radius=None,
        approx_backend="graph",
        preference=None,
    )

    with pytest.raises(SystemExit):
        validate_cli_arguments(args)


def test_validate_cli_arguments_accepts_radius_graph_with_graph_backend():
    args = SimpleNamespace(
        n_neighbors=16,
        neighbor_search="faiss",
        n_representatives=None,
        method="radius-graph",
        mode="approximate",
        threshold=None,
        radius=1.0,
        approx_backend="graph",
        preference=None,
    )

    validate_cli_arguments(args)


def test_validate_cli_arguments_rejects_neighbor_search_without_graph_backend():
    args = SimpleNamespace(
        n_neighbors=16,
        neighbor_search="faiss",
        n_representatives=None,
        method="hierarchical",
        mode="approximate",
        threshold=None,
        radius=None,
        approx_backend="dense",
        preference=None,
    )

    with pytest.raises(SystemExit):
        validate_cli_arguments(args)
