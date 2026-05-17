import pytest
from tileforge.backend.metal.ir import GPUModule, GPUFunction, GPUBlock, GPUOperation, GPUOpType, GPUValue
from tileforge.backend.metal.verifier import GPUIRVerifier
from tileforge.frontend.errors import IRVerificationError
from tileforge.ir.types import I1, I32, F32, PointerType, TensorType

def test_valid_module():
    module = GPUModule()
    func = GPUFunction("test_func", [])
    module.functions.append(func)
    
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    ret_op = GPUOperation(GPUOpType.RETURN)
    block.operations.append(ret_op)
    
    verifier = GPUIRVerifier()
    verifier.verify_module(module)

def test_empty_function():
    module = GPUModule()
    func = GPUFunction("test_func", [])
    module.functions.append(func)
    
    verifier = GPUIRVerifier()
    with pytest.raises(IRVerificationError, match="has no blocks"):
        verifier.verify_module(module)

def test_missing_terminator():
    module = GPUModule()
    func = GPUFunction("test_func", [])
    module.functions.append(func)
    
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    add_op = GPUOperation(GPUOpType.ADD)
    block.operations.append(add_op)
    
    verifier = GPUIRVerifier()
    with pytest.raises(IRVerificationError, match="must end with a terminator operation"):
        verifier.verify_module(module)

def test_invalid_cond_br():
    verifier = GPUIRVerifier()
    func = GPUFunction("test", [])
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    op = GPUOperation(GPUOpType.COND_BR)
    block.operations.append(op)
    module = GPUModule()
    module.functions.append(func)
    with pytest.raises(IRVerificationError, match="requires condition operand"):
        verifier.verify_module(module)

    block.operations.clear()
    op = GPUOperation(GPUOpType.COND_BR, operands=[GPUValue("c", I32)])
    block.operations.append(op)
    with pytest.raises(IRVerificationError, match="condition must be i1"):
        verifier.verify_module(module)

def test_invalid_global_load():
    verifier = GPUIRVerifier()
    func = GPUFunction("test", [])
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    op = GPUOperation(GPUOpType.GLOBAL_LOAD, operands=[GPUValue("ptr", PointerType(F32))])
    block.operations.append(op)
    op2 = GPUOperation(GPUOpType.RETURN)
    block.operations.append(op2)
    module = GPUModule()
    module.functions.append(func)
    with pytest.raises(IRVerificationError, match="requires pointer and offset operands"):
        verifier.verify_module(module)

def test_verifier_negative_invalid_branch_target():
    verifier = GPUIRVerifier()
    func = GPUFunction("test", [])
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    invalid_block = GPUBlock("external_block")
    
    op = GPUOperation(GPUOpType.BR, operands=[], successors=[invalid_block])
    block.operations.append(op)
    
    module = GPUModule()
    module.functions.append(func)
    with pytest.raises(IRVerificationError, match="invalid branch target block"):
        verifier.verify_module(module)

def test_verifier_negative_tiled_dot_invalid_tile():
    verifier = GPUIRVerifier()
    func = GPUFunction("test", [])
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    v1 = GPUValue("a", F32)
    v2 = GPUValue("b", F32)
    res = GPUValue("out", F32)
    
    op = GPUOperation(GPUOpType.TILED_DOT, operands=[v1, v2], results=[res], attributes={"BM": -16})
    block.operations.append(op)
    block.operations.append(GPUOperation(GPUOpType.RETURN))
    
    module = GPUModule()
    module.functions.append(func)
    with pytest.raises(IRVerificationError, match="Tile dimensions must be positive"):
        verifier.verify_module(module)

def test_verifier_multiple_tiled_dot_rejected():
    verifier = GPUIRVerifier()
    func = GPUFunction("test", [])
    block = GPUBlock("entry")
    func.blocks.append(block)
    
    v1 = GPUValue("a", F32)
    v2 = GPUValue("b", F32)
    res1 = GPUValue("out1", F32)
    res2 = GPUValue("out2", F32)
    
    op1 = GPUOperation(GPUOpType.TILED_DOT, operands=[v1, v2], results=[res1], attributes={"BM": 16, "BN": 16, "BK": 16})
    op2 = GPUOperation(GPUOpType.TILED_DOT, operands=[v1, v2], results=[res2], attributes={"BM": 16, "BN": 16, "BK": 16})
    block.operations.append(op1)
    block.operations.append(op2)
    block.operations.append(GPUOperation(GPUOpType.RETURN))
    
    module = GPUModule()
    module.functions.append(func)
    with pytest.raises(IRVerificationError, match="maximum 1 allowed per kernel"):
        verifier.verify_module(module)
