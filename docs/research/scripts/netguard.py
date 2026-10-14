"""Block + log every non-loopback network attempt made from Python.

Import this FIRST. Records attempted hosts in ATTEMPTS and prints them at exit.
"""
import atexit
import socket
import sys
import traceback

ATTEMPTS = []
_LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0", None, ""}

_orig_getaddrinfo = socket.getaddrinfo
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _caller():
    for fr in reversed(traceback.extract_stack()[:-2]):
        f = fr.filename
        if "netguard" in f:
            continue
        if "site-packages" in f:
            return f.split("site-packages/")[-1] + ":" + str(fr.lineno)
    return "?"


def _getaddrinfo(host, *a, **k):
    h = host.decode() if isinstance(host, bytes) else host
    if h not in _LOCAL:
        ATTEMPTS.append(("dns", h, _caller()))
        raise socket.gaierror(f"netguard: blocked DNS for {h}")
    return _orig_getaddrinfo(host, *a, **k)


def _check(addr):
    if isinstance(addr, tuple) and addr and isinstance(addr[0], str):
        ip = addr[0]
        if not (ip.startswith("127.") or ip == "::1"):
            ATTEMPTS.append(("connect", f"{ip}:{addr[1]}", _caller()))
            raise ConnectionRefusedError(f"netguard: blocked connect to {addr}")


def _connect(self, addr):
    _check(addr)
    return _orig_connect(self, addr)


def _connect_ex(self, addr):
    _check(addr)
    return _orig_connect_ex(self, addr)


socket.getaddrinfo = _getaddrinfo
socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex


@atexit.register
def _report():
    print(f"[netguard] blocked non-loopback attempts: {len(ATTEMPTS)}", file=sys.stderr)
    seen = set()
    for a in ATTEMPTS:
        if a not in seen:
            seen.add(a)
            print(f"[netguard]   {a}", file=sys.stderr)
