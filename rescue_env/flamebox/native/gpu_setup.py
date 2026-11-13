"""Local GPU readiness and build diagnostic. Does not install untrusted software or bypass Windows policy."""
from __future__ import annotations
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build import find_nvcc, cuda_compiler_environment, compiler_environment


def check_command(argv,timeout=15):
    try:
        r=subprocess.run(argv,text=True,errors='replace',capture_output=True,timeout=timeout)
        return r.returncode,(r.stdout+'\n'+r.stderr).strip()
    except (OSError,subprocess.TimeoutExpired) as e:
        return -1,str(e)


def inspect():
    print('FlameBox NVIDIA GPU readiness',flush=True)
    status={}
    smi=shutil.which('nvidia-smi')
    if not smi and sys.platform=='win32':
        for candidate in (
            Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'NVIDIA Corporation'/'NVSMI'/'nvidia-smi.exe',
            Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'/'nvidia-smi.exe'):
            if candidate.is_file():smi=str(candidate);break
    rc,out=check_command([smi or 'nvidia-smi','-L'])
    status['driver']=rc==0
    print('NVIDIA driver/GPU:',out if rc==0 else 'NOT DETECTED ('+out[:200]+')')
    nvcc=find_nvcc()
    status['nvcc']=bool(nvcc)
    print('CUDA Toolkit nvcc:',nvcc or 'NOT FOUND')
    if nvcc:
        rc,output=check_command([nvcc,'--version'])
        print('Toolkit version:',output.splitlines()[-1] if rc==0 and output else output[:200])
    if sys.platform=='win32':
        # Detect MSVC even before CUDA is installed: avoid prompting users to
        # reinstall Visual Studio Build Tools they already have.
        base_env=compiler_environment(nvcc) if nvcc else os.environ.copy()
        env=cuda_compiler_environment(nvcc or '',base_env)
        status['msvc']=bool(shutil.which('cl.exe',path=env.get('PATH')))
        print('MSVC host compiler:', 'ready' if status['msvc'] else 'NOT DETECTED')
    suffix='flamebox_gpu.dll' if sys.platform=='win32' else ('libflamebox_gpu.dylib' if sys.platform=='darwin' else 'libflamebox_gpu.so')
    plugin=HERE/suffix
    status['plugin']=plugin.is_file()
    print('FlameBox compiled GPU plugin:',str(plugin) if plugin.is_file() else 'NOT INSTALLED')
    if plugin.exists():
        try:
            lib=ctypes.CDLL(str(plugin));lib.flamebox_cuda_available.restype=ctypes.c_int
            lib.flamebox_cuda_device.restype=ctypes.c_char_p
            device_count=lib.flamebox_cuda_available()
            status['device_test']=device_count>0
            print('CUDA plugin/device:',lib.flamebox_cuda_device().decode(errors='replace') if device_count else 'No CUDA-capable device')
        except OSError as e:
            print('CUDA plugin load error:',str(e));status['device_test']=False
    if not status['nvcc']:
        print('Install official CUDA Toolkit >= 12.8 for RTX 50-series from https://developer.nvidia.com/cuda-downloads')
        if sys.platform=='win32':print('Optional Windows package manager: winget install -e --id Nvidia.CUDA')
    if sys.platform=='win32' and status['nvcc'] and not status.get('msvc'):
        print('Install Visual Studio C++ Build Tools (Desktop development with C++ workload).')
    if sys.platform=='win32':
        print('Official install commands (run only if you approve the downloads):')
        print('  winget install -e --id Nvidia.CUDA')
        print('  winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"')
        print('For an existing Build Tools installation, use Visual Studio Installer -> Modify -> C++ build tools.')
    print('Linux HPC: request a GPU compute node and load the cluster CUDA module BEFORE building.')
    return status


def install_prerequisites(status):
    """Opt-in official toolchain installer. Never bypass security or install silently."""
    if sys.platform!='win32':
        print('Dependency installation must be managed using your Linux distribution or HPC modules.')
        return False
    winget=shutil.which('winget')
    if not winget:
        print('winget is unavailable. Use the official CUDA and Visual Studio download links above.')
        return False
    commands=[]
    if not status.get('msvc'):
        commands.append(('Visual Studio 2022 Build Tools + C++ workload (large download)',
                         [winget,'install','-e','--id','Microsoft.VisualStudio.2022.BuildTools',
                          '--override','--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended']))
    if not status['nvcc']:
        commands.append(('NVIDIA CUDA Toolkit (large download)',
                         [winget,'install','-e','--id','Nvidia.CUDA']))
    if not commands:return True
    for label,args in commands:
        print('\nInstall',label,'using:',subprocess.list2cmdline(args),flush=True)
        try:
            consent=input('Type INSTALL to allow this third-party download and installation, or press Enter to skip: ').strip()
        except (EOFError,KeyboardInterrupt):
            return False
        if consent!='INSTALL':
            print('Skipped. Nothing installed by FlameBox.')
            return False
        try:
            done=subprocess.run(args,check=False)
        except OSError as exc:
            print('Installer failed to start:',exc)
            return False
        if done.returncode!=0:
            print('Installer exited with code',done.returncode,'. Check winget/Windows installer diagnostics.')
            return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true',help='Compile CUDA plugin using installed CPU library and test real NVIDIA GPU')
    parser.add_argument('--install',action='store_true',help='Request explicit approval before installing prerequisites with winget; then build')
    args=parser.parse_args()
    status=inspect()
    if args.install:
        if not install_prerequisites(status):
            return 2
        status=inspect()
    if args.build or args.install:
        if not status['nvcc'] or (sys.platform=='win32' and not status.get('msvc')):
            print('Cannot compile CUDA until the missing prerequisites above are installed.')
            return 2
        if not status['driver']:
            print('No accessible NVIDIA GPU/driver. On HPC build on an allocated GPU compute node.')
            return 2
        if not (HERE/('sim_native.dll' if sys.platform=='win32' else 'libsim_native.dylib' if sys.platform=='darwin' else 'libsim_native.so')).is_file():
            print('CPU reference library is absent. Build the CPU library first: python build.py')
            return 2
        rc,out=check_command([sys.executable,str(HERE/'build.py'),'--cuda-only'],timeout=600)
        print(out)
        if rc:return rc
        result=inspect()
        return 0 if result.get('device_test') else 3
    return 0 if status.get('device_test') else 2

if __name__=='__main__':sys.exit(main())
