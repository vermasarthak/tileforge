import numpy as np

from tileforge.backend.metal.mps_fallback import MPSGraphPartitioner
from tileforge.ir.function import Function
from tileforge.ir.operation import Operation, OpType
from tileforge.ir.types import F32, VOID, SymInt, TensorType


def test_symint_type():
    sym_m = SymInt("M")
    sym_n = SymInt("N")
    tensor_t = TensorType((sym_m, sym_n), F32)
    assert str(tensor_t) == "tensor<MxNxf32>"


def test_mps_graph_partitioner():
    partitioner = MPSGraphPartitioner()
    func = Function(name="test_fn", args=[], return_type=VOID)
    b = func.create_block(name="entry")
    b.append_operation(Operation(OpType.PROGRAM_ID, operands=[], results=[], attributes={"axis": 0}))
    b.append_operation(Operation(OpType.CONSTANT, operands=[], results=[], attributes={"value": 1.0}))

    subgraphs, meta = partitioner.partition_function(func)
    assert meta["total_subgraphs"] == 1
    assert meta["native_metal_count"] == 1
    assert meta["mps_fallback_count"] == 0

    res = partitioner.execute_fallback_op("matmul", [np.eye(2), np.eye(2)])
    np.testing.assert_allclose(res, np.eye(2))

    # Test activations and softmax fallback
    arr = np.array([0.0, 1.0, -1.0])
    np.testing.assert_allclose(partitioner.execute_fallback_op("sigmoid", [arr]), 1.0 / (1.0 + np.exp(-arr)))
    np.testing.assert_allclose(partitioner.execute_fallback_op("relu", [arr]), np.array([0.0, 1.0, 0.0]))
    softmax_res = partitioner.execute_fallback_op("softmax", [arr])
    np.testing.assert_allclose(np.sum(softmax_res), 1.0)
