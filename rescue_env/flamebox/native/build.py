"""Compile the FlameBox C++17 kernel and record actionable compiler diagnostics.

Native builds are not signed. On managed Windows computers Application Control
may prevent loading the resulting DLL; no security settings are modified here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
LOG = HERE / 'BUILD_NATIVE.log'
IS_WINDOWS = sys.platform == 'win32'
IS_MAC = sys.platform == 'darwin'
OUTPUT = 'sim_native.dll' if IS_WINDOWS else ('libsim_native.dylib' if IS_MAC else 'libsim_native.so')


def find_compiler():
    override = os.environ.get('CXX')
    if override:
        return shutil.which(override) or (str(Path(override)) if Path(override).is_file() else None)
    # Prefer UCRT64 on Windows over an unrelated GCC earlier in PATH.
    candidates = ([r'C:\msys64\ucrt64\bin\g++.exe',
                   r'C:\msys64\mingw64\bin\g++.exe'] if IS_WINDOWS else [])
    candidates += ['g++', 'clang++', 'cl']
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        if Path(candidate).is_file():
            return str(Path(candidate))
    return None


class BuildError(RuntimeError):
    pass


def compiler_environment(compiler):
    """Windows GCC needs its own UCRT64/MINGW64 runtime directory on PATH.

    MSYS2 g++.exe starts without this, but its cc1plus.exe child may fail
    silently before compiling even the simplest C++ file.
    """
    env = os.environ.copy()
    if IS_WINDOWS:
        compiler_bin = str(Path(compiler).resolve().parent)
        env['PATH'] = compiler_bin + os.pathsep + env.get('PATH', '')
    return env


def register_build_dll_dir(compiler):
    """Retain OS DLL directory search handle until the load check finishes."""
    if IS_WINDOWS and hasattr(os, 'add_dll_directory'):
        return os.add_dll_directory(str(Path(compiler).resolve().parent))
    return None


def run_stage(title, args, output, env=None):
    output.append(f'\n--- {title} ---\n$ {subprocess.list2cmdline([str(x) for x in args])}\n')
    print(f'{title}:', flush=True)
    try:
        completed = subprocess.run([str(x) for x in args], capture_output=True,
                                   text=True, errors='replace', timeout=180, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        output.append(f'Unable to execute command: {type(exc).__name__}: {exc}\n')
        raise BuildError(f'{title} could not execute: {exc}') from exc
    for name, text in [('stdout', completed.stdout), ('stderr', completed.stderr)]:
        if text:
            output.append(f'{name}:\n{text}\n')
            print(text[-6000:], file=sys.stderr if name == 'stderr' else sys.stdout, flush=True)
    output.append(f'Exit code: {completed.returncode}\n')
    if completed.returncode != 0:
        hint = ('The compiler did not provide diagnostics. Verify that Windows Application '
                'Control permits g++.exe and its subprocesses, and inspect Windows Security '
                'or Event Viewer for blocked components.' if not (completed.stderr or completed.stdout)
                else 'See the diagnostic output above.')
        raise BuildError(f'{title} failed (exit {completed.returncode}). {hint}')
    return completed


def validate_built_library(candidate, compiler, report, env):
    """Check the staged library from a fresh interpreter before installing it.

    A subprocess avoids holding a loaded DLL open while TemporaryDirectory is
    cleaned up on Windows, and verifies the same DLL discovery as Brain workers.
    """
    checker = (
        "import os,sys,ctypes; "
        "h=os.add_dll_directory(sys.argv[2]) if os.name=='nt' else None; "
        "lib=ctypes.CDLL(sys.argv[1]); getattr(lib,'sim_new'); "
        "print('Native library load/export validation: PASSED')"
    )
    try:
        run_stage('Load and verify compiled library',
                  [sys.executable, '-c', checker, str(candidate),
                   str(Path(compiler).resolve().parent)], report, env=env)
    except BuildError as exc:
        explanation = str(exc)
        if '4551' in ''.join(report[-5:]):
            explanation += ' Windows Application Control may be blocking an unsigned DLL.'
        raise BuildError(explanation) from exc


def compile_and_verify(compiler, is_msvc, use_openmp, temp, report, env):
    """Compile AND link together as one strategy; never reuse an OpenMP object
    for a serial fallback (it contains unresolved GOMP symbols).
    """
    mode = 'OpenMP multicore' if use_openmp else 'portable single-thread'
    built = temp / OUTPUT
    if built.exists():
        built.unlink()
    if is_msvc:
        args = [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/LD']
        if use_openmp:
            args.append('/openmp')
        args += [str(HERE / 'sim.cpp'), '/link', '/OUT:' + str(built)]
        run_stage('Compile and link (' + mode + ')', args, report, env=env)
    else:
        obj = temp / ('flamebox_omp.o' if use_openmp else 'flamebox_serial.o')
        flags = ['-std=c++17', '-O3', '-DNDEBUG']
        if not IS_WINDOWS:
            flags += ['-fPIC']
        if use_openmp:
            flags += ['-fopenmp']
        run_stage('Compile C++ source (' + mode + ')',
                  [compiler, *flags, '-c', str(HERE / 'sim.cpp'), '-o', str(obj)],
                  report, env=env)
        link = [compiler, '-dynamiclib' if IS_MAC else '-shared', str(obj)]
        if use_openmp:
            link.append('-fopenmp')
        if IS_WINDOWS:
            link += ['-static-libgcc', '-static-libstdc++']
        link += ['-o', str(built)]
        run_stage('Link native library (' + mode + ')', link, report, env=env)
    if not built.exists() or built.stat().st_size == 0:
        raise BuildError(mode + ': compiler returned success but produced no library')
    return built


def install_and_validate(built, compiler, report, env):
    """Validate the DLL only at its permanent path, not from %%TEMP%%.

    Windows security software may enforce a different policy for temporary
    executable images. Restore the previous binary if installation/load fails.
    This is not a Smart App Control bypass: unsigned binaries may still be blocked.
    """
    target = HERE / OUTPUT
    backup = target.with_name(target.name + '.previous-build')
    if backup.exists():
        backup.unlink()
    try:
        if target.exists():
            # Move the old library out of the way, preserving it for rollback.
            target.replace(backup)
        shutil.copy2(built, target)
        validate_built_library(target, compiler, report, env)
    except (BuildError, OSError) as exc:
        try:
            if target.exists():
                target.unlink()
            if backup.exists():
                backup.replace(target)
        except OSError as restore_exc:
            report.append(f'ROLLBACK FAILED: {restore_exc}\n')
        raise BuildError(f'Native library install/load rejected: {exc}') from exc
    else:
        if backup.exists():
            try: backup.unlink()
            except OSError: report.append('WARNING: previous build backup could not be removed.\n')
    return target



def find_nvcc():
    """Discover CUDA even when the installer did not add nvcc to PATH."""
    override=os.environ.get('NVCC')
    if override:
        return shutil.which(override) or (override if Path(override).is_file() else None)
    resolved=shutil.which('nvcc')
    if resolved:return resolved
    candidates=[]
    for name in ('CUDA_PATH','CUDA_HOME'):
        base=os.environ.get(name)
        if base:candidates.append(Path(base)/'bin'/('nvcc.exe' if IS_WINDOWS else 'nvcc'))
    if IS_WINDOWS:
        programfiles=os.environ.get('ProgramFiles',r'C:\Program Files')
        base=Path(programfiles)/'NVIDIA GPU Computing Toolkit'/'CUDA'
        if base.is_dir():
            candidates.extend(sorted(base.glob('v*/bin/nvcc.exe'),reverse=True))
    else:
        candidates.extend([Path('/usr/local/cuda/bin/nvcc'),Path('/opt/cuda/bin/nvcc')])
    return next((str(x) for x in candidates if x.is_file()),None)


def cuda_compiler_environment(nvcc, env):
    """nvcc on Windows needs a supported MSVC host environment, not MSYS2 GCC."""
    if not IS_WINDOWS or shutil.which('cl.exe',path=env.get('PATH')):
        return env
    install=Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))
    vswhere=install/'Microsoft Visual Studio'/'Installer'/'vswhere.exe'
    if not vswhere.is_file():return env
    try:
        found=subprocess.run([str(vswhere),'-latest','-products','*',
             '-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
             '-property','installationPath'],capture_output=True,text=True,timeout=15)
        if found.returncode:return env
        vs=Path(found.stdout.strip())
        bat=vs/'VC'/'Auxiliary'/'Build'/'vcvars64.bat'
        if not bat.is_file():return env
        # Windows cmd.exe explicitly initializes INCLUDE/LIB/PATH for nvcc.
        # cmd quoting must preserve spaces in Visual Studio's installation path.
        line=f'call "{bat}" >nul && set'
        result=subprocess.run(['cmd.exe','/d','/s','/c',line],capture_output=True,
                              text=True,errors='replace',timeout=25,env=env)
        if result.returncode:return env
        updated=env.copy()
        for row in result.stdout.splitlines():
            if '=' in row and not row.startswith('='):
                key,value=row.split('=',1)
                updated[key]=value
        return updated
    except (OSError,subprocess.TimeoutExpired):
        return env


def build_cuda_plugin(report, *, require_cuda=False):
    """Optional CUDA backend; no nvcc/driver means a valid CPU-only build.

    GPU source compiles into a separately loaded plugin rather than linking
    CUDA into the CPU reference binary. On Linux HPC nodes, load the CUDA
    environment module before invoking this command.
    """
    target = HERE / ('flamebox_gpu.dll' if IS_WINDOWS else
                     'libflamebox_gpu.dylib' if IS_MAC else 'libflamebox_gpu.so')
    nvcc = find_nvcc()
    if not nvcc:
        report.append('CUDA plugin: nvcc not found; GPU backend unavailable.\n')
        if require_cuda:
            raise BuildError('CUDA requested but nvcc was not found. Install/load NVIDIA CUDA Toolkit and rerun build.py --require-cuda.')
        return False
    env = cuda_compiler_environment(nvcc,compiler_environment(nvcc))
    with tempfile.TemporaryDirectory(prefix='flamebox-cuda-') as directory:
        candidate = Path(directory) / target.name
        # A100 (sm_80) gets native SASS; newer NVIDIA devices can JIT
        # forward-compatible compute_80 PTX, including RTX 50 / Blackwell.
        # Native Blackwell sm_120 code is optional via FLAMEBOX_CUDA_ARCH=120
        # with CUDA 12.8+ and an appropriate host compiler.
        args = [nvcc, '-std=c++17', '-O3', '-shared',
                '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_80,code=compute_80']
        extra_arch=os.environ.get('FLAMEBOX_CUDA_ARCH','').strip()
        if not extra_arch:
            # Detect native GPU architecture on the device used for compilation.
            # Keep sm_80 SASS + compute_80 PTX for A100/HPC portability.
            smi=shutil.which('nvidia-smi')
            if IS_WINDOWS and not smi:
                for candidate in (
                    Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'/'nvidia-smi.exe',
                    Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'NVIDIA Corporation'/'NVSMI'/'nvidia-smi.exe'):
                    if candidate.is_file():smi=str(candidate);break
            if smi:
                try:
                    info=subprocess.run([smi,'--query-gpu=compute_cap','--format=csv,noheader'],
                                        capture_output=True,text=True,timeout=8)
                    if info.returncode==0 and info.stdout.strip():
                        cap=info.stdout.strip().splitlines()[0].strip()
                        major,minor=cap.split('.')[:2]
                        native_arch=int(major)*10+int(minor)
                        if 80<native_arch<=120:extra_arch=str(native_arch)
                except (OSError,ValueError,subprocess.TimeoutExpired):
                    pass
        if extra_arch:
            if not extra_arch.isdigit() or not 80 <= int(extra_arch) <= 120:
                raise BuildError('FLAMEBOX_CUDA_ARCH must be e.g. 86, 89, 90 or 120')
            if extra_arch!='80':
                args.append(f'-gencode=arch=compute_{extra_arch},code=sm_{extra_arch}')
        if not IS_WINDOWS:
            args += ['-Xcompiler=-fPIC']
        args += [str(HERE / 'sim_cuda.cu'), '-o', str(candidate)]
        try:
            run_stage('Compile optional CUDA physics plugin', args, report, env=env)
            if not candidate.exists() or candidate.stat().st_size == 0:
                raise BuildError('nvcc did not produce a CUDA shared library')
            # Stage a replacement safely. If the new plugin fails its device
            # test, recover the last working GPU plugin rather than deleting it.
            backup = target.with_name(target.name + '.previous-build')
            if backup.exists():
                backup.unlink()
            if target.exists():
                target.replace(backup)
            shutil.copy2(candidate, target)
            # Do not announce an operational CUDA backend from compilation
            # alone. Execute a conservation/bounds smoke test on the real GPU.
            run_stage('Run CUDA device kernel self-test',
                      [sys.executable, str(HERE / 'gpu_selftest.py')], report, env=env)
            if backup.exists():
                backup.unlink()
            report.append(f'CUDA plugin built and device-tested: {target}\n')
            print('CUDA plugin built:', target, flush=True)
            return True
        except (BuildError, OSError) as exc:
            try:
                target.unlink(missing_ok=True)
                if 'backup' in locals() and backup.exists():
                    backup.replace(target)
            except OSError:
                pass
            report.append(f'CUDA plugin not available: {exc}\n')
            print('WARNING: CUDA plugin compilation failed; CPU solver remains available:', exc, flush=True)
            if require_cuda:
                raise BuildError('Required CUDA compilation failed: ' + str(exc)) from exc
            return False


def cuda_only_build():
    """Compile/verify GPU support without rebuilding an already working CPU library.

    This is intended for GPU setup and automatic bootstrapping after installing
    the CUDA Toolkit. The GPU regression test needs the CPU reference installed.
    """
    report = [f'FlameBox CUDA-only build, {time.strftime("%Y-%m-%d %H:%M:%S")}\n',
              f'Platform: {platform.platform()}\n',
              f'CUDA nvcc: {find_nvcc() or "NOT FOUND"}\n']
    try:
        if not (HERE / OUTPUT).exists():
            raise BuildError(f'CPU library missing: {HERE / OUTPUT}. First run build.py without --cuda-only.')
        build_cuda_plugin(report, require_cuda=True)
        return 0
    except (BuildError, OSError) as exc:
        report.append('CUDA BUILD FAILED: ' + str(exc) + '\n')
        print('ERROR:', exc, file=sys.stderr, flush=True)
        print('Build diagnostics:', LOG, file=sys.stderr, flush=True)
        return 1
    finally:
        LOG.write_text(''.join(report), encoding='utf-8')


def build(*, force_serial=False, require_openmp=False, require_cuda=False):
    report = [f'FlameBox native build, {time.strftime("%Y-%m-%d %H:%M:%S")}\n',
              f'Python: {sys.version.split()[0]}\n',
              f'Platform: {platform.platform()}\n',
              f'Target: {HERE / OUTPUT}\n']
    try:
        compiler = find_compiler()
        report.append(f'Compiler: {compiler or "NOT FOUND"}\n')
        if not compiler:
            raise BuildError('No C++17 compiler found. Install MSYS2 UCRT64 g++ or MSVC Build Tools.')
        env = compiler_environment(compiler)
        report.append(f"Compiler runtime PATH prefix: {Path(compiler).resolve().parent if IS_WINDOWS else '-'}\n")
        is_msvc = Path(compiler).name.lower() in ('cl', 'cl.exe')
        run_stage('Compiler check', [compiler, '/?' if is_msvc else '--version'], report, env=env)

        with tempfile.TemporaryDirectory(prefix='flamebox-cpp-') as tempdir:
            temp = Path(tempdir)
            openmp_eligible = not force_serial and (is_msvc or not IS_MAC)
            mode_used = 'serial'
            if openmp_eligible:
                try:
                    built = compile_and_verify(compiler, is_msvc, True, temp, report, env)
                    install_and_validate(built, compiler, report, env)
                    mode_used = 'OpenMP'
                except BuildError as exc:
                    report.append('OpenMP compile/link/load failed; retrying from source with NO OpenMP.\n')
                    report.append(f'OpenMP failure: {exc}\n')
                    print('WARNING: OpenMP unavailable; rebuilding native C++ without OpenMP.', flush=True)
                    if IS_WINDOWS and 'lgomp' in str(exc) + ''.join(report[-12:]):
                        print('Missing MSYS2 OpenMP library. Optional repair in MSYS2 UCRT64:',flush=True)
                        print('  pacman -S --needed mingw-w64-ucrt-x86_64-libgomp',flush=True)
                        report.append('MSYS2 UCRT64 OpenMP package: mingw-w64-ucrt-x86_64-libgomp\n')
                    if 'WinError 4551' in str(exc) or 'security policy' in str(exc).lower():
                        raise BuildError('Windows Application Control rejected the binary. See CodeIntegrity event 3077 and BUILD_NATIVE.log. Signing is required when Windows refuses unknown unsigned code.') from exc
                    if require_openmp:
                        raise BuildError('OpenMP was required, but its compilation, link, or load failed.') from exc
                    built = compile_and_verify(compiler, is_msvc, False, temp, report, env)
                    install_and_validate(built, compiler, report, env)
            else:
                if require_openmp:
                    raise BuildError('OpenMP required but disabled by options or unsupported on this build.')
                built = compile_and_verify(compiler, is_msvc, False, temp, report, env)
                install_and_validate(built, compiler, report, env)

            # Installed library passed a fresh-process load check at final path.
            report.append(f'Library installed: {HERE / OUTPUT}\n')
            report.append(f'Execution backend: {mode_used}\n')
            print('Native library created:', HERE / OUTPUT, flush=True)
            print('Build mode:', 'OpenMP multicore' if mode_used == 'OpenMP'
                  else 'C++ native (single-thread: OpenMP not linked)', flush=True)
            if mode_used != 'OpenMP':
                print('NOTE: C++ physics works, but a single simulation will not use OpenMP threads.',flush=True)
            build_cuda_plugin(report, require_cuda=require_cuda)
    except BuildError as exc:
        report.append('BUILD FAILED: ' + str(exc) + '\n')
        print('ERROR:', exc, file=sys.stderr, flush=True)
        print('Full diagnostics:', LOG, file=sys.stderr, flush=True)
        if IS_WINDOWS:
            print('For Smart App Control blocks: Event Viewer > Applications and Services Logs > Microsoft > Windows > CodeIntegrity > Operational > Event 3077.', file=sys.stderr, flush=True)
            print('Windows can block unsigned locally built DLLs. FlameBox will not disable Windows security or attempt to bypass this policy.', file=sys.stderr, flush=True)
        return 1
    finally:
        LOG.write_text(''.join(report), encoding='utf-8')
    print('Build diagnostics:', LOG, flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument('--serial', action='store_true', help='Build native C++ without OpenMP')
    backend.add_argument('--require-openmp', action='store_true', help='Fail instead of falling back to single-thread C++')
    parser.add_argument('--require-cuda', action='store_true',help='Require nvcc and build CUDA backend; fail if absent')
    parser.add_argument('--cuda-only', action='store_true', help='Compile and validate only GPU plugin, without rebuilding CPU')
    args = parser.parse_args()
    if args.cuda_only and (args.serial or args.require_openmp):
        parser.error('--cuda-only must not be combined with --serial or --require-openmp')
    sys.exit(cuda_only_build() if args.cuda_only else build(
        force_serial=args.serial, require_openmp=args.require_openmp, require_cuda=args.require_cuda))
