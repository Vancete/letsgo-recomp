# scroll_speed.py <dir>...: the scene's vertical scroll between consecutive window shots (phase correlation), px per second
# (dirs from tools/winseries.ps1: s<k>.png + times.txt). Its gray() / shift() also compare game shots (frame_N.png) taken
# at the same game time at different frame rates - how the world's speed was measured at 30 / 60 / 120 (CHANGELOG).
import sys, os
import numpy as np
from PIL import Image

def gray(p):
    a = np.asarray(Image.open(p).convert('L'), dtype=np.float64)
    a = a.copy()
    a[230:540, 520:780] = a.mean()  # (the player and Pikachu: the camera follows them, they stay put)
    a = a[60:-120, 100:-100]  # (the bars / the corner icon out)
    return (a - a.mean()) * np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]

def shift(a, b):
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    if y > r.shape[0] // 2: y -= r.shape[0]
    if x > r.shape[1] // 2: x -= r.shape[1]
    return y, x

for d in (sys.argv[1:] if __name__ == "__main__" else []):
    t = dict(map(int, l.split()) for l in open(os.path.join(d, 'times.txt')) if l.strip())
    ks = sorted(t)
    rates = []
    for k0, k1 in zip(ks, ks[1:]):
        dy, dx = shift(gray(os.path.join(d, f's{k0}.png')), gray(os.path.join(d, f's{k1}.png')))
        v = dy / ((t[k1] - t[k0]) / 1000)
        rates.append(v)
        print(f'{os.path.basename(d)} {k0}->{k1}: dy {dy:4d} dx {dx:3d} in {t[k1]-t[k0]} ms = {v:6.1f} px/s')
    print(f'{os.path.basename(d)}: median {np.median(rates):.1f} px/s')
