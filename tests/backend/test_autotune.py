from tileforge.autotune import Config, autotune


def test_autotune_decorator():
    configs = [
        Config({"BLOCK_M": 16, "BLOCK_N": 16}),
        Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ]

    executed_configs = []

    @autotune(configs=configs, key=["M", "N"])
    def mock_kernel(A, B, C, M=16, N=16, BLOCK_M=16, BLOCK_N=16):
        executed_configs.append((BLOCK_M, BLOCK_N))
        return BLOCK_M * BLOCK_N

    res = mock_kernel([1], [1], [1], M=16, N=16)
    assert res > 0
    assert len(executed_configs) > 0
