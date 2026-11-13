"""CUDA driver/kernel smoke test, not a substitute for scientific validation.

Run after nvcc compilation on a GPU compute node. No Orb/server required.
"""
import ctypes as C
import platform
from pathlib import Path
import numpy as np

native = Path(__file__).resolve().parent
filename = ('flamebox_gpu.dll' if platform.system() == 'Windows' else
            'libflamebox_gpu.dylib' if platform.system() == 'Darwin' else 'libflamebox_gpu.so')
if platform.system() == 'Windows' and hasattr(__import__('os'), 'add_dll_directory'):
    _cuda_path = __import__('os').add_dll_directory(str(native))
lib = C.CDLL(str(native / filename))
ptr = C.POINTER(C.c_float)
arrs = C.POINTER(ptr)
lib.flamebox_cuda_available.restype = C.c_int
lib.flamebox_cuda_device.restype = C.c_char_p
lib.flamebox_cuda_error.restype = C.c_char_p
lib.flamebox_cuda_new.argtypes = [C.c_int, C.c_int]
lib.flamebox_cuda_new.restype = C.c_void_p
lib.flamebox_cuda_free.argtypes = [C.c_void_p]
lib.flamebox_cuda_step.argtypes = [C.c_void_p, arrs, arrs, C.POINTER(C.c_uint8),
                                  C.POINTER(C.c_uint8), ptr, C.c_int, C.c_int, C.c_bool]
lib.flamebox_cuda_step.restype = C.c_int
if lib.flamebox_cuda_available() < 1:
    raise RuntimeError('No working NVIDIA CUDA driver/device visible to process')
H, W = 8, 12
N = H * W
fields = [np.zeros(N, np.float32) for _ in range(12)]
coeffs = [np.zeros(N, np.float32) for _ in range(9)]
fields[0][:] = 1.0
fields[1][:] = .1
fields[1][35] = .7
fields[10][:] = .03
fields[10][35] = .21
fields[9][:] = .3
fields[3][:] = .8
for k in (0,1,2,3):
    coeffs[k][:] = .012
for k in (6,7):
    coeffs[k][:] = 1.0
coeffs[8][:] = 1.0
p = np.zeros(48, np.float32)
p[0] = .08
p[27] = 1.0
p[28] = 1.0
p[10] = 0.0
material = np.zeros(N,np.uint8)
debris = np.zeros(N,np.uint8)
smoke_before = float(np.sum(fields[1], dtype=np.float64))
mass_before = float(np.sum(fields[10], dtype=np.float64))
obj = lib.flamebox_cuda_new(H,W)
if not obj:
    raise RuntimeError('GPU context creation failed: ' + lib.flamebox_cuda_error().decode())
try:
    fp = (ptr * 12)(*(a.ctypes.data_as(ptr) for a in fields))
    cp = (ptr * 9)(*(a.ctypes.data_as(ptr) for a in coeffs))
    result = lib.flamebox_cuda_step(obj,fp,cp,
              material.ctypes.data_as(C.POINTER(C.c_uint8)),
              debris.ctypes.data_as(C.POINTER(C.c_uint8)),
              p.ctypes.data_as(ptr),3,4,True)
    if result:
        raise RuntimeError('GPU step failed: ' + lib.flamebox_cuda_error().decode())
    for a in fields:
        assert np.all(np.isfinite(a)), 'CUDA produced a non-finite field'
    assert np.min(fields[1]) >= -1e-5 and np.max(fields[1]) <= 1 + 1e-5
    assert np.min(fields[10]) >= -1e-5 and np.max(fields[10] - fields[1]) < 1e-5
    smoke_after = float(np.sum(fields[1], dtype=np.float64))
    mass_after = float(np.sum(fields[10], dtype=np.float64))
    assert abs(smoke_after-smoke_before) < 1e-4, (smoke_after,smoke_before)
    assert abs(mass_after-mass_before) < 1e-4, (mass_after,mass_before)
    print('GPU smoke test passed on',lib.flamebox_cuda_device().decode())
    print('Smoke mass error:',smoke_after-smoke_before,
          'Flammable mass error:',mass_after-mass_before)
finally:
    lib.flamebox_cuda_free(obj)

# Coupled smoke/fire/oxygen regression on the same native environment.
# This detects GPU-vs-CPU wiring errors missed by the small conservation test.
# It does NOT establish physical-model validity against experimental data.
import os
import sys
sys.path.insert(0,str(native.parent))
from flamebox import Environment
for resolution in (3,8):
    states=[]
    for solver in ('parallel','cuda'):
        os.environ['FLAMEBOX_SOLVER']=solver
        e=Environment(size=8,resolution=resolution,victims=0,ignitions=2,seed=397)
        # Include active smoke transport and combustible-smoke mixing.
        e.smoke[:]=.35
        e.smoke_flammable_mass[:]=.07
        e.tick(2)
        states.append({k:getattr(e,k).copy() for k in
             ('oxygen','smoke','fire','fuel','damage','smoke_flammable_mass')})
    print('Coupled CPU/GPU numerical comparison at resolution',resolution)
    for name,cpu in states[0].items():
        gpu=states[1][name]
        if not np.all(np.isfinite(gpu)) or np.min(gpu)<-1.e-5 or np.max(gpu)>1.0001:
            raise AssertionError(f'CUDA {name}: nonfinite or out-of-bounds numerical result')
        delta=np.abs(cpu-gpu)
        maxd=float(np.max(delta));meand=float(np.mean(delta))
        print(f'  {name}: mean_abs_diff={meand:.8g} max_abs_diff={maxd:.8g}')
        if meand>.025:
            raise AssertionError(f'CUDA {name}: large mean mismatch against CPU parallel solver')
    if np.any(states[1]['smoke_flammable_mass']>states[1]['smoke']+1.e-5):
        raise AssertionError('CUDA: combustible smoke exceeds total smoke')
print('Coupled GPU smoke/fire regression: PASSED (bounded short-run numerical agreement)')
