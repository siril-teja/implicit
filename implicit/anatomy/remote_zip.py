"""Read single members of a remote ZIP over HTTP range requests (stdlib only).

Lets us pull one subject (a few MB) out of a multi-GB dataset archive.
"""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP with a small block cache."""

    def __init__(self, url, block=1 << 17):
        self.url, self.block, self.pos = url, block, 0
        self.cache: dict[int, bytes] = {}
        self.size = self._size()

    def _get(self, start, end, tries=5):
        for i in range(tries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"server ignored Range (HTTP {r.status})")
                    return r.read()
            except (OSError, IOError):
                if i == tries - 1:
                    raise
                time.sleep(2 ** (i + 1))

    def _size(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return int(r.headers["Content-Range"].split("/")[1])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if n > 4 * self.block:  # large read: fetch directly
            data = self._get(self.pos, self.pos + n - 1)
        else:
            out, p, end = [], self.pos, self.pos + n
            while p < end:
                k = p // self.block
                if k not in self.cache:
                    s = k * self.block
                    self.cache[k] = self._get(s, min(s + self.block, self.size) - 1)
                chunk = self.cache[k]
                lo = p - k * self.block
                take = chunk[lo:lo + (end - p)]
                out.append(take)
                p += len(take)
            data = b"".join(out)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 16))
