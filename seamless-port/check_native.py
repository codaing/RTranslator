"""Reject ELF LOAD segments aligned below 16 KiB; verify ARM64-only libraries."""
import struct
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as z:
    libs = [n for n in z.namelist() if n.startswith("lib/") and n.endswith(".so")]
    assert libs and all(n.startswith("lib/arm64-v8a/") for n in libs), libs
    for n in libs:
        data = z.read(n)
        assert data[:6] == b"\x7fELF\x02\x01", n
        offset = struct.unpack_from("<Q", data, 32)[0]
        size, count = struct.unpack_from("<HH", data, 54)
        alignments = []
        for i in range(count):
            fields = struct.unpack_from("<IIQQQQQQ", data, offset+i*size)
            if fields[0] == 1:
                alignments.append(fields[-1])
        assert alignments and min(alignments) >= 16384, (n, alignments)
        print(n, alignments, "PASS")
