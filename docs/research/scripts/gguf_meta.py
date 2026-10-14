"""Parse GGUF metadata KV pairs from a (possibly truncated) GGUF file header.
Usage: python3 -I gguf_meta.py <file> [key_filter_substring ...]
Arrays longer than 8 elements are summarised (length + first items).
"""
import struct
import sys

T = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def main():
    path = sys.argv[1]
    filters = sys.argv[2:]
    data = open(path, 'rb').read()
    off = 0

    def rd(fmt):
        nonlocal off
        v = struct.unpack_from('<' + fmt, data, off)
        off += struct.calcsize('<' + fmt)
        return v[0]

    def rstr():
        nonlocal off
        n = rd('Q')
        s = data[off:off + n].decode('utf-8', 'replace')
        off += n
        return s

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et = rd('I')
            n = rd('Q')
            items = []
            for i in range(n):
                v = rval(et)
                if i < 8:
                    items.append(v)
            return ('ARRAY', n, items)
        return rd(T[t])

    magic = data[:4]
    off = 4
    ver = rd('I')
    nt = rd('Q')
    nkv = rd('Q')
    print('magic', magic, 'version', ver, 'tensors', nt, 'kv', nkv)
    for _ in range(nkv):
        try:
            k = rstr()
            t = rd('I')
            v = rval(t)
        except struct.error:
            print('TRUNCATED at offset', off)
            break
        if filters and not any(f in k for f in filters):
            continue
        if isinstance(v, str) and len(v) > 300:
            v = v[:300] + '...'
        print(k, '=', v)


main()
