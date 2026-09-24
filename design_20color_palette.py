#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""تصميم لوحة 20 لوناً: كل فئة لون مستقل تماماً (لا عائلات لونية متكررة)"""
import colorsys
import numpy as np

# ---------- تحويلات لونية ----------
def hex2rgb(h):
    return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255.0

def srgb2lab(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T
    wp = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / wp
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return np.array([L, a, b])

def dE(l1, l2):
    return float(np.sqrt(((l1 - l2) ** 2).sum()))

def hsv2hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return "#%02X%02X%02X" % (round(r*255), round(g*255), round(b*255))

# ---------- الخلفيات ----------
LAND = srgb2lab(hex2rgb("#f6f2e8"))
OCEAN = srgb2lab(hex2rgb("#d4e4f0"))
print(f"Lab الأرض: {LAND.round(1)} | المحيط: {OCEAN.round(1)}")

# ---------- المجموعة المرشحة ----------
pool = []
for h in range(0, 360, 10):
    for s in (0.55, 0.75, 1.0):
        for v in (0.55, 0.7, 0.85, 1.0):
            hx = hsv2hex(h, s, v)
            lab = srgb2lab(hex2rgb(hx))
            C = float(np.hypot(lab[1], lab[2]))
            if 32 <= lab[0] <= 90 and C >= 25:          # مرئي ومشبع
                if dE(lab, LAND) < 22 or dE(lab, OCEAN) < 18:
                    continue                             # قريب من الخلفيات
                pool.append((hx, lab, h))
print(f"عدد المرشحين بعد الترشيح: {len(pool)}")

# ---------- اختيار جشع (نقطة الأبعد) ----------
# البذرة: أعلى تباعد عن الخلفيات
seed = max(pool, key=lambda c: min(dE(c[1], LAND), dE(c[1], OCEAN)))
sel = [seed]
rest = [c for c in pool if c[0] != seed[0]]
while len(sel) < 20 and rest:
    best, bd = None, -1
    for c in rest:
        m = min(dE(c[1], s[1]) for s in sel)
        if m > bd:
            bd, best = m, c
    sel.append(best)
    rest.remove(best)

print(f"\n=== الألوان العشرون المختارة (مرتبة بزاوية الصبغة) ===")
sel.sort(key=lambda c: c[2])
for hx, lab, h in sel:
    C = np.hypot(lab[1], lab[2])
    print(f"  {hx}  hue={h:3d}°  L={lab[0]:5.1f}  C={C:5.1f}  dE(land)={dE(lab,LAND):5.1f}")

# أضعف الأزواج
pairs = []
for i in range(len(sel)):
    for j in range(i + 1, len(sel)):
        pairs.append((dE(sel[i][1], sel[j][1]), sel[i][0], sel[j][0]))
pairs.sort()
print(f"\nأضعف 5 أزواج (كل الأزواج):")
for d, a, b in pairs[:5]:
    print(f"  {a} ↔ {b}: ΔE = {d:.1f}")
print(f"أدنى ΔE بين أي زوج: {pairs[0][0]:.1f}")

# ---------- ترتيب متناوب (نصف العجلة بين الجيران) ----------
order = []
idx = list(range(len(sel)))
for k in range(10):
    order += [idx[k], idx[k + 10]]
print("\n=== الترتيب المتناوب (كل لونين متجاورين متقابلان تقريباً في العجلة) ===")
seq = [sel[i] for i in order]
for n, (hx, lab, h) in enumerate(seq):
    print(f"  الفئة {n+1:2d}: {hx}  (hue {h}°)")
adj = [dE(seq[i][1], seq[i+1][1]) for i in range(19)]
print(f"\nأدنى ΔE بين فئتين متجاورتين في المفتاح: {min(adj):.1f} | متوسط: {np.mean(adj):.1f}")
print(f"الفئة 1 مقابل الأرض: {dE(seq[0][1], LAND):.1f} | مقابل المحيط: {dE(seq[0][1], OCEAN):.1f}")
print("\nLEVELS =", "[0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 140]")
print("RAIN_COLORS = [")
for n in range(0, 20, 2):
    row = ", ".join(f'"{seq[n][0]}"' if n + k < 20 else "" for k in (0, 1))
    row = ", ".join(f'"{seq[m][0]}"' for m in (n, n + 1) if m < 20)
    print(f'    "{seq[n][0]}", "{seq[n+1][0]}",' if n + 1 < 20 else f'    "{seq[n][0]}",')
print("]")
