from examples.benchmark_decode import print_memory_table, print_read_table


def test_benchmark_decode_tables_print(capsys) -> None:
    print_memory_table()
    print_read_table()

    output = capsys.readouterr().out

    assert "KV cache memory" in output
    assert "Dense vs simplified DSA/CSA decode reads" in output
    assert "MHA" in output
    assert "2604.2x" in output
