"""Native Apple Metal Runtime via Objective-C / ctypes FFI bridge.

Handles MTLDevice, MTLCommandQueue, MTLBuffer, MTLComputePipelineState, command buffers,
encoders, grid dispatch, and synchronous execution on Apple Silicon M1 GPU.
"""

from __future__ import annotations
import ctypes
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# Load System Libraries
libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")

# Basic FFI Types
libobjc.objc_getClass.restype = ctypes.c_void_p
libobjc.objc_getClass.argtypes = [ctypes.c_char_p]

libobjc.sel_registerName.restype = ctypes.c_void_p
libobjc.sel_registerName.argtypes = [ctypes.c_char_p]

def get_sel(name: str) -> ctypes.c_void_p:
    return libobjc.sel_registerName(name.encode("utf-8"))

# Metal Specific Types
class MTLSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint64), ("height", ctypes.c_uint64), ("depth", ctypes.c_uint64)]

MTLCreateSystemDefaultDevice = metal.MTLCreateSystemDefaultDevice
MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
MTLCreateSystemDefaultDevice.argtypes = []

# Hardcoded CFUNCTYPEs for ARM64 Objective-C method dispatch to avoid segfaults
msg_send_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))
msg_send_void = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))
msg_send_char_p = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))

msg_send_string = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(("objc_msgSend", libobjc))
msg_send_new_lib = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p))(("objc_msgSend", libobjc))
msg_send_id_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))
msg_send_new_pipeline = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(("objc_msgSend", libobjc))
msg_send_new_buffer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)(("objc_msgSend", libobjc))
msg_send_void_id = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", libobjc))
msg_send_set_buffer = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)(("objc_msgSend", libobjc))
msg_send_set_bytes = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)(("objc_msgSend", libobjc))
msg_send_dispatch = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize)(("objc_msgSend", libobjc))


def ns_string(text: str) -> int:
    cls_str = libobjc.objc_getClass(b"NSString")
    sel_str = get_sel("stringWithUTF8String:")
    return msg_send_string(cls_str, sel_str, text.encode("utf-8"))


class MetalDevice:
    """Encapsulates Apple Metal MTLDevice and MTLCommandQueue."""
    def __init__(self):
        self.device = MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Failed to create Metal default device")

        # Create MTLCommandQueue
        self.command_queue = msg_send_id(self.device, get_sel("newCommandQueue"))
        if not self.command_queue:
            raise RuntimeError("Failed to create Metal command queue")

    def get_name(self) -> str:
        name_obj = msg_send_id(self.device, get_sel("name"))
        utf8_str = msg_send_char_p(name_obj, get_sel("UTF8String"))
        return utf8_str.decode("utf-8") if utf8_str else "Apple Metal Device"


class MetalCompiler:
    """Compiles MSL source code into MTLComputePipelineState via runtime Metal compiler."""
    def __init__(self, device: MetalDevice):
        self.device = device.device

    def compile_source(self, msl_source: str, kernel_name: str) -> int:
        source_ns = ns_string(msl_source)

        err_ptr = ctypes.c_void_p(0)
        # device.newLibraryWithSource:options:error:
        library = msg_send_new_lib(self.device, get_sel("newLibraryWithSource:options:error:"), source_ns, 0, ctypes.byref(err_ptr))
        if not library:
            raise RuntimeError(f"MSL Compilation Failed. Source:\n{msl_source}")

        # library.newFunctionWithName:
        func_name_ns = ns_string(kernel_name)
        function = msg_send_id_id(library, get_sel("newFunctionWithName:"), func_name_ns)
        if not function:
            raise RuntimeError(f"Function '{kernel_name}' not found in compiled Metal library")

        # device.newComputePipelineStateWithFunction:error:
        pipeline = msg_send_new_pipeline(self.device, get_sel("newComputePipelineStateWithFunction:error:"), function, ctypes.byref(err_ptr))
        if not pipeline:
            raise RuntimeError(f"Failed to create Metal compute pipeline for '{kernel_name}'")

        return pipeline


class MetalBuffer:
    """Encapsulates MTLBuffer for zero-copy/shared host-device memory."""
    def __init__(self, device: MetalDevice, size_bytes: int):
        self.device = device.device
        self.size_bytes = size_bytes
        # MTLResourceStorageModeShared = 0 << 4 = 0
        self.buf = msg_send_new_buffer(self.device, get_sel("newBufferWithLength:options:"), size_bytes, 0)
        if not self.buf:
            raise RuntimeError(f"Failed to allocate Metal buffer of size {size_bytes} bytes")

    def contents(self) -> ctypes.c_void_p:
        return msg_send_id(self.buf, get_sel("contents"))

    def upload_numpy(self, np_arr: np.ndarray) -> None:
        c_ptr = self.contents()
        ctypes.memmove(c_ptr, np_arr.ctypes.data, np_arr.nbytes)

    def download_numpy(self, np_arr: np.ndarray) -> None:
        c_ptr = self.contents()
        ctypes.memmove(np_arr.ctypes.data, c_ptr, np_arr.nbytes)


class MetalRuntime:
    """Executes compiled Metal pipeline state synchronously on Apple Silicon GPU."""
    def __init__(self):
        self.device = MetalDevice()
        self.compiler = MetalCompiler(self.device)

    def dispatch(
        self,
        pipeline: int,
        grid: Tuple[int, ...],
        args: List[Any],
        threads_per_threadgroup: Tuple[int, ...] = (256, 1, 1),
    ) -> None:
        cmd_buffer = msg_send_id(self.device.command_queue, get_sel("commandBuffer"))
        encoder = msg_send_id(cmd_buffer, get_sel("computeCommandEncoder"))

        msg_send_void_id(encoder, get_sel("setComputePipelineState:"), pipeline)

        metal_buffers: List[MetalBuffer] = []
        host_arrs: List[Tuple[int, np.ndarray, MetalBuffer]] = []

        for idx, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                buf = MetalBuffer(self.device, arg.nbytes)
                buf.upload_numpy(arg)
                metal_buffers.append(buf)
                host_arrs.append((idx, arg, buf))
                # encoder.setBuffer:offset:atIndex:
                msg_send_set_buffer(encoder, get_sel("setBuffer:offset:atIndex:"), buf.buf, 0, idx)
            elif isinstance(arg, (int, np.integer)):
                c_val = ctypes.c_int32(int(arg))
                # encoder.setBytes:length:atIndex:
                msg_send_set_bytes(encoder, get_sel("setBytes:length:atIndex:"), ctypes.byref(c_val), 4, idx)
            elif isinstance(arg, (float, np.floating)):
                c_val = ctypes.c_float(float(arg))
                msg_send_set_bytes(encoder, get_sel("setBytes:length:atIndex:"), ctypes.byref(c_val), 4, idx)
            else:
                raise TypeError(f"Unsupported Metal kernel argument type: {type(arg)}")

        # Format MTLSize structures for grid and threadgroups
        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1

        tx = threads_per_threadgroup[0] if len(threads_per_threadgroup) > 0 else 1
        ty = threads_per_threadgroup[1] if len(threads_per_threadgroup) > 1 else 1
        tz = threads_per_threadgroup[2] if len(threads_per_threadgroup) > 2 else 1

        grid_size = MTLSize(gx, gy, gz)
        threadgroup_size = MTLSize(tx, ty, tz)

        # Dispatch
        msg_send_dispatch(encoder, get_sel("dispatchThreadgroups:threadsPerThreadgroup:"), grid_size, threadgroup_size)

        msg_send_void(encoder, get_sel("endEncoding"))
        msg_send_void(cmd_buffer, get_sel("commit"))
        msg_send_void(cmd_buffer, get_sel("waitUntilCompleted"))

        # Copy updated buffer results back to host numpy arrays
        for idx, arr, buf in host_arrs:
            buf.download_numpy(arr)
