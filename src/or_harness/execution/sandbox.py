"""Trusted Linux namespace bootstrap for SafePythonExecutor.

Only runtime distributions, declared input FILES and a fresh output directory
are mounted. A read-only mount enforces the same boundary for Python, native
extensions and solver file APIs. No host cwd, /proc, memory bank or credentials
are exposed. Setup errors fail closed, never fall back to AST-only execution.

This is process confinement, not a VM: the kernel and installed runtime are
trusted. Metadata such as stat() is not a confidentiality boundary.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import sys
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path


class SandboxAccessError(PermissionError):
    """A script attempted an operation outside its documented file contract."""


def _audit_paths(runtime):
    def audit(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path, mode, flags = args
        path = os.fsdecode(path)
        if ".." in Path(path).parts:
            raise SandboxAccessError("parent traversal is outside the execution contract: " + path)
        target = os.path.realpath(path)
        write = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        in_work = target == "/work" or target.startswith("/work/")
        in_runtime = any(target == p or target.startswith(p + "/") for p in runtime)
        writable = target == "/work/result.json" or target.startswith("/work/scratch/")
        if write and not writable:
            if target != "/dev/null":
                raise SandboxAccessError("writes require the isolated output workspace: " + path)
        elif not write and not (in_work or in_runtime):
            raise SandboxAccessError("read requires a declared task input or runtime file: " + path)
    sys.addaudithook(audit)


def runtime_paths():
    candidates = [sysconfig.get_path(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")]
    # Conda extensions need sibling shared libraries; do not expose the rest
    # of the environment prefix (which can contain user work and credentials).
    candidates += [str(Path(sys.base_prefix) / "lib"), "/usr/lib", "/lib", "/lib64"]
    paths = []
    for value in sorted(set(candidates), key=len):
        path = Path(value)
        if path.is_dir() and not any(path == p or p in path.parents for p in paths):
            paths.append(path)
    for value in ("/etc/ld.so.cache", "/etc/localtime", "/dev/null", "/dev/urandom",
                  "/proc/self/maps", "/proc/cpuinfo", sys.executable):
        if Path(value).exists():
            paths.append(Path(value))
    return [str(p) for p in paths]


def command(config_path):
    unshare = shutil.which("unshare")
    if sys.platform != "linux" or not unshare:
        raise OSError("executor isolation requires Linux unshare and mount/network namespaces")
    return [unshare, "--user", "--map-root-user", "--mount", "--net",
            sys.executable, "-I", str(Path(__file__).resolve()), str(config_path)]


@lru_cache(maxsize=1)
def isolation_probe():
    """One tiny synthetic probe per process; no task data or solver solve."""
    from or_harness.execution.executor import SafePythonExecutor
    with tempfile.TemporaryDirectory(prefix="orx-isolation-probe-") as directory:
        work = Path(directory)
        script = work / "solve.py"
        script.write_text("import json\njson.dump({'status':'optimal','objective_value':0},open('result.json','w'))\n")
        result = SafePythonExecutor(timeout_seconds=5).run(script, work, "python")
    return {"backend": "linux-user-mount-net-namespaces+seccomp",
            "available": result.status == "optimal", "probe_scope": "synthetic isolated script",
            "failure": result.normalized_error or None,
            "requirements": "Linux user/mount/network namespaces, unshare, libseccomp.so.2; fail closed if unavailable",
            "limits": "trusts host kernel/runtime; no VM, metadata secrecy or protection from concurrent host mutation"}


def _mount(libc, source, target, *, readonly=False):
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target.mkdir(exist_ok=True)
    else:
        target.touch(exist_ok=True)
    if libc.mount(os.fsencode(source), os.fsencode(target), None, 4096, None):  # MS_BIND
        raise OSError(ctypes.get_errno(), "bind mount failed: " + str(source))
    if readonly:
        # MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID. Bind mounts of
        # /dev/null and /dev/urandom intentionally retain device semantics.
        _readonly(libc, target)


def _readonly(libc, target):
    inherited = os.statvfs(target).f_flag
    flags = 32 | 4096 | 1 | 2 | (inherited & (4 | 8 | 1024 | 2048))
    if inherited & 4096:  # ST_RELATIME -> MS_RELATIME
        flags |= 1 << 21
    if libc.mount(None, os.fsencode(target), None, flags, None):
        raise OSError(ctypes.get_errno(), "read-only remount failed: " + str(target))


def _syscall_filter(lib):
    # The default allows numerical computing and threads. Denied operations
    # include all process creation except CLONE_THREAD, executable launches,
    # sockets, namespace/mount mutation and cross-process access. clone3 gets
    # ENOSYS so glibc can use the filtered clone call for solver threads.
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise OSError("seccomp_init failed")
    try:
        blocked = ("fork vfork execve execveat socket socketpair connect bind listen accept accept4 "
                   "mount umount2 pivot_root chroot unshare setns ptrace process_vm_readv "
                   "process_vm_writev pidfd_getfd open_by_handle_at name_to_handle_at "
                   "fsopen fsmount fsconfig move_mount open_tree mount_setattr "
                   "io_uring_setup bpf perf_event_open userfaultfd keyctl add_key request_key "
                   "reboot kexec_load kexec_file_load init_module finit_module delete_module "
                   "swapon swapoff sethostname setdomainname kill tkill pidfd_send_signal "
                   "rt_sigqueueinfo rt_tgsigqueueinfo mknod mknodat "
                   "shmget shmat msgget msgsnd msgrcv semget semop semtimedop").split()
        for name in blocked + ["clone3"]:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number == -1:
                continue  # syscall does not exist on this architecture
            action = 0x00050000 | (errno.ENOSYS if name == "clone3" else errno.EPERM)
            if lib.seccomp_rule_add(context, action, number, 0):
                raise OSError("seccomp rule failed: " + name)

        class Compare(ctypes.Structure):
            _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                        ("mask", ctypes.c_uint64), ("value", ctypes.c_uint64)]

        # SCMP_CMP_MASKED_EQ: deny clone when CLONE_THREAD is absent.
        compare = Compare(0, 7, 0x10000, 0)
        number = lib.seccomp_syscall_resolve_name(b"clone")
        if number < 0 or lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM,
                                               number, 1, compare):
            raise OSError("seccomp clone rule failed")
        compare = Compare(0, 1, os.getpid(), 0)  # SCMP_CMP_NE, own tgid only
        number = lib.seccomp_syscall_resolve_name(b"tgkill")
        if number >= 0 and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM,
                                               number, 1, compare):
            raise OSError("seccomp tgkill rule failed")
        if lib.seccomp_load(context):
            raise OSError("seccomp_load failed")
    finally:
        lib.seccomp_release(context)


def _bootstrap(config):
    libc = ctypes.CDLL(None, use_errno=True)
    # Load before chroot; /usr/lib is mounted read-only for extension loading.
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    root = Path(config["root"])
    if libc.mount(None, b"/", None, 16384 | (1 << 18), None):  # PRIVATE | REC
        raise OSError(ctypes.get_errno(), "cannot make mounts private")
    _mount(libc, root, root)
    for runtime in config["runtime_paths"]:
        _mount(libc, runtime, root / runtime.lstrip("/"), readonly=runtime != "/dev/null")
    # Some native numerical runtimes locate their loader through these own-
    # process entries. Never mount /proc itself, fd/, environ, or another PID.
    own_exe = root / "proc" / "self" / "exe"
    own_exe.parent.mkdir(parents=True, exist_ok=True)
    own_exe.symlink_to(sys.executable)
    _mount(libc, config["work"], root / "work")
    _mount(libc, config["script"], root / "work" / "solve.py", readonly=True)
    inputs = root / "work" / "inputs"
    inputs.mkdir(exist_ok=True)
    # Mount input hierarchy separately so the parent itself is also read-only.
    _mount(libc, inputs, inputs)
    for resource in config["inputs"]:
        _mount(libc, resource["source"], inputs / resource["path"], readonly=True)
    _readonly(libc, inputs)
    # Only the result inode and bounded scratch are writable, including from
    # native code. All other jail directories are read-only mounts. The
    # scratch mount has an aggregate 64 MiB limit, independent of RLIMIT_FSIZE.
    _mount(libc, Path(config["work"]) / "result.json", root / "work" / "result.json")
    scratch = root / "work" / "scratch"
    if libc.mount(b"tmpfs", os.fsencode(scratch), b"tmpfs", 2 | 4 | 8, b"size=64m,mode=700"):
        raise OSError(ctypes.get_errno(), "cannot mount bounded scratch")
    _readonly(libc, root / "work")
    _readonly(libc, root)
    os.chroot(root)
    os.chdir("/work")
    # No open descriptor may point at a host directory. Popen closes all
    # inherited descriptors except stdio and the short-lived receipt pipe.
    # No new privileges; namespace-root's capabilities cannot mutate mounts
    # after seccomp is installed. The namespace contains no /proc or devices
    # other than null/urandom, and no executable can be launched.
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "no_new_privs failed")
    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class CapData(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]
    header, data = CapHeader(0x20080522, 0), (CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)):
        raise OSError(ctypes.get_errno(), "cannot drop namespace capabilities")
    _syscall_filter(seccomp)
    _audit_paths(config["runtime_paths"])


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    receipt = config["receipt_fd"]
    try:
        _bootstrap(config)
        # Read and compile only AFTER isolation. The parent supplied a frozen
        # script copy; no code in the caller's workspace executes at startup.
        source = Path("/work/solve.py").read_bytes()
        code = compile(source, "/work/solve.py", "exec")
    except BaseException as exc:
        os.write(receipt, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())
        os.close(receipt)
        return 125
    os.write(receipt, b'{"ready":true}')
    os.close(receipt)
    sys.argv = ["/work/solve.py"]
    exec(code, {"__name__": "__main__", "__file__": "/work/solve.py"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
