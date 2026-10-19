"""disasm.py <addr> [count] | disasm.py bl <target> | disasm.py ref <addr>: the game's main (build/xrefs-main.pkl of xrefs.py).

  disasm.py <addr> [n]   n instructions from addr (default 40)
  disasm.py bl <target>  every bl / b to target, with its function start (the closest preceding stp x29, x30 / sub sp)
  disasm.py ref <addr>   adrp + add / ldr / str pairs that form addr (a global's readers and writers)
"""
import bisect
import re
import sys

from xrefs import load

d = load()
ins = d['ins']
addrs = [a for a, _, _ in ins]


def at(a):
    return bisect.bisect_left(addrs, a)


def func_start(i):
    for j in range(i, max(i - 4000, 0), -1):
        m, op = ins[j][1], ins[j][2]
        if (m == 'stp' and 'x29, x30' in op) or (m == 'sub' and op.startswith('sp, sp')):
            # walk back over the prologue's other stores
            k = j
            while k > 0 and ins[k - 1][1] in ('stp', 'str', 'sub') and 'sp' in ins[k - 1][2]:
                k -= 1
            return ins[k][0]
    return None


if sys.argv[1] == 'bl':
    t = int(sys.argv[2], 16)
    for i, (a, m, op) in enumerate(ins):
        if m in ('bl', 'b') and op == '#0x%x' % t:
            print('%x %s %s   (in %s)' % (a, m, op, '%x' % (func_start(i) or 0)))
elif sys.argv[1] == 'ref':
    t = int(sys.argv[2], 16)
    page = t & ~0xfff
    for i, (a, m, op) in enumerate(ins):
        if m == 'adrp' and op.endswith('#0x%x' % page):
            reg = op.split(',')[0]
            for k in range(i + 1, min(i + 12, len(ins))):
                a2, m2, op2 = ins[k]
                if reg in op2 and ('#0x%x' % (t - page)) in op2:
                    print('%x %s %s ; %x %s %s   (in %x)' % (a, m, op, a2, m2, op2, func_start(i) or 0))
                    break
else:
    a = int(sys.argv[1], 16)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    i = at(a)
    for a2, m, op in ins[i:i + n]:
        print('%x  %-8s %s' % (a2, m, op))
