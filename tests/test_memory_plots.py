from attention_forge.viz.memory_plots import memory_curve_points, sparse_read_points


def test_memory_curve_points_use_expected_variants() -> None:
    points = memory_curve_points([4096])

    assert points["MHA"] == [2.0]
    assert points["MQA"] == [0.0625]
    assert points["GQA-8"] == [0.5]
    assert points["MLA-512"] == [0.125]
    assert points["DSA dense-storage"] == [2.0]


def test_sparse_read_points_show_dense_growth_and_sparse_cap() -> None:
    points = sparse_read_points([4096, 1_000_000])

    assert points["Dense"] == [4096, 1_000_000]
    assert points["DSA/CSA reads"] == [320, 384]
