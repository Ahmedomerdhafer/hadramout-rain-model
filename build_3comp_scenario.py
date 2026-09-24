#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  بناء سيناريو ثلاثي المكوّنات (كلي / حملي / طبقي) لأحدث دورة GFS تشغيلية
  build_3comp_scenario.py YYYYMMDD CC     (مثال: python3 build_3comp_scenario.py 20260914 00)
===============================================================================
  السلسلة (مُتحقَّق منها بإعادة إنتاج دورة 13/18Z بدقة ≤ 0.17 ملم على كل الشبكة):
    1) تنزيل APCP + ACPCP التراكمية 0–240 ساعة من ملف f240 عبر byte-range
       (فهرس .idx يحدد موضع الرسالة — ~1MB لكل رسالة بدل 500MB)
    2) QDM (Cannon 2015) على منظومة v17-HR1:
         cnt    = عدد مرجعيات < x0                      (24 دورة مرجعية)
         q      = clip((cnt+0.5)/(n+1), 1/(n+1), n/(n+1))   ← حدود Weibull
         ref_q  = استيفاء خطي لـ q على مواضع (i+0.5)/n للمرجعيات المرتبة
         g      = γ.ppf(q, α=0.4845, scale=mu17/α)  ، و = 0 حيث mu17 = 0
         scen   = clip(g + x0 − ref_q, 0, ∞)
    3) الحملي: نفس السلسلة بمرجعيات ACPCP ومناخ mu17_conv (= mu17 × f_ref،
       النسبة الحملية الإقليمية 82.6%) مع سقف scen_conv ≤ scen_tot
    4) الطبقي (مطر السحب المنخفضة) = الكلي − الحملي (بالبناء)
===============================================================================
"""
import sys, os, re
import numpy as np
import requests
import eccodes
from scipy.stats import gamma

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CACHE = "gfs_cache/op"
V17 = "gfs_cache/v17replay"
LAT0, LAT1, LON0, LON1 = 11.0, 21.0, 44.0, 56.0   # نطاق النطاق (تصاعدي)
STATIONS = {
    'Mukalla': (49.12, 14.53), 'Ash Shihr': (49.60, 14.76),
    'Al-Dhabba Port': (49.50, 14.70), 'Shuhayr': (49.42, 14.65),
    'Wadi Huwayrah': (49.35, 14.82), 'Seiyun (Wadi)': (48.78, 15.93),
    'Tarim': (49.00, 16.05), 'Dawan (Wadi)': (48.45, 15.00),
    'Al-Ghaydah': (52.19, 16.21),
}


def parse_idx(text):
    """تحليل فهرس f240 — السطور بصيغتين: 'n:n:byte:d=..' أو 'n:byte:d=..'"""
    entries = []
    for ln in text.splitlines():
        f = ln.split(':')
        try:
            di = next(i for i, x in enumerate(f) if x.startswith('d='))
        except StopIteration:
            continue
        byte = next((int(x) for x in reversed(f[:di]) if x.strip().isdigit()), None)
        if byte is None:
            continue
        var = f[di + 1] if len(f) > di + 1 else ''
        lev = f[di + 2] if len(f) > di + 2 else ''
        desc = f[di + 3] if len(f) > di + 3 else ''
        entries.append((byte, var, lev, desc))
    return entries


def fetch_message(ymd, cc, var, pat, idx_path):
    """تنزيل رسالة GRIB واحدة عبر byte-range وإرجاع قيمها (721×1440)"""
    url = f"{BASE}/gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f240"
    entries = parse_idx(open(idx_path).read())
    hits = [(b, v, l, d) for (b, v, l, d) in entries
            if v == var and l == 'surface' and re.search(pat, d)]
    assert len(hits) == 1, f"{var}: توقعت رسالة واحدة، وجدت {len(hits)}: {hits}"
    start = hits[0][0]
    later = [b for (b, _, _, _) in entries if b > start]
    end = (min(later) - 1) if later else int(
        requests.head(url, timeout=60).headers['Content-Length']) - 1
    blob = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=300).content
    tmp = os.path.join(CACHE, f"msg_{ymd}{cc}_{var}.bin")
    open(tmp, 'wb').write(blob)
    with open(tmp, 'rb') as fh:
        g = eccodes.codes_grib_new_from_file(fh)
        short = eccodes.codes_get(g, 'shortName')
        step = eccodes.codes_get(g, 'stepRange')
        vals = eccodes.codes_get_values(g).reshape(721, 1440)
        eccodes.codes_release(g)
    os.remove(tmp)
    assert short in ('tp', 'acpcp') and step == '0-240', f"{var}: {short}/{step}؟!"
    print(f"  ✅ {var}: {short} {step} ({len(blob):,} بايت)")
    return vals


def crop_flip(vals):
    """قص النطاق من الشبكة العالمية (صفوف تنازلية العرض) ثم قلبها تصاعدياً"""
    r0 = int(round((90.0 - LAT1) / 0.25))   # 276  (lat 21)
    r1 = int(round((90.0 - LAT0) / 0.25))   # 316  (lat 11)
    c0 = int(round(LON0 / 0.25))            # 176  (lon 44)
    c1 = int(round(LON1 / 0.25))            # 224  (lon 56)
    return vals[r0:r1 + 1, c0:c1 + 1][::-1, :].astype(np.float32)  # ← تصاعدي


def qdm(x0, refs_sorted, mu, alpha):
    """QDM المُتحقَّق منه — يعيد إنتاج دورة 18Z بدقة ≤ 0.17 ملم"""
    n = refs_sorted.shape[0]
    cnt = (refs_sorted < x0[None]).sum(axis=0)
    q = np.clip((cnt + 0.5) / (n + 1), 1.0 / (n + 1), n / (n + 1))
    p = (np.arange(n) + 0.5) / n
    qq = np.clip(q, p[0], p[-1])
    i1 = np.clip(np.searchsorted(p, qq, side='right'), 1, n - 1)
    i0 = i1 - 1
    w = (qq - p[i0]) / (p[i1] - p[i0])
    s0 = np.take_along_axis(refs_sorted, i0[None], 0)[0]
    s1 = np.take_along_axis(refs_sorted, i1[None], 0)[0]
    ref_q = s0 * (1 - w) + s1 * w
    with np.errstate(all='ignore'):
        g = np.where(mu > 0, gamma.ppf(q, alpha, scale=mu / alpha), 0.0)
    return np.clip(np.nan_to_num(g) + x0 - ref_q, 0, None)


def main(ymd, cc):
    os.makedirs(CACHE, exist_ok=True)
    idx_path = f"{CACHE}/gfs_{ymd}{cc}_f240.idx"
    if not os.path.exists(idx_path):
        r = requests.get(f"{BASE}/gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f240.idx", timeout=120)
        r.raise_for_status()
        open(idx_path, 'w').write(r.text)
    print(f"📡 دورة {ymd}/{cc}Z — تنزيل الرسائل التراكمية 0–240 ساعة:")
    # "0-10 day acc fcst" (تسمية حديثة) أو "0-240 hour acc fcst" (قديمة)
    x0t = crop_flip(fetch_message(ymd, cc, 'APCP', r'0-(240 hour|10 day) acc', idx_path))
    x0c = crop_flip(fetch_message(ymd, cc, 'ACPCP', r'0-(240 hour|10 day) acc', idx_path))
    x0s = np.clip(x0t - x0c, 0, None).astype(np.float32)
    over = int((x0c > x0t + 1e-6).sum())
    if over:
        print(f"  ⚠️ {over} نقطة ACPCP>APCP (تقريب عددي) — قُصّت")
        x0c = np.minimum(x0c, x0t)

    # المكوّنات المناخية v17-HR1 + العينة المرجعية (24 دورة)
    mu17 = np.load(f"{V17}/scenario_20260912.npz")["mu17"].astype(np.float64)
    mu17c = np.load(f"{V17}/scenario_conv_2026091306.npz")["mu17_conv"].astype(np.float64)
    alpha = float(np.load(f"{V17}/scenario_20260912.npz")["alpha"])
    lat = np.round(np.arange(LAT0, LAT1 + 1e-9, 0.25), 2)
    lon = np.round(np.arange(LON0, LON1 + 1e-9, 0.25), 2)
    refs = np.stack([np.load(p)[::-1, :] for p in sorted(
        __import__('glob').glob(f"{V17}/v16ref/*.npy"))]).astype(np.float64)
    crefs = np.stack([np.load(p)[::-1, :] for p in sorted(
        __import__('glob').glob(f"{V17}/convref/*.npy"))]).astype(np.float64)
    n = refs.shape[0]

    scen_tot = qdm(x0t.astype(np.float64), np.sort(refs, 0), mu17, alpha)
    scen_conv = np.minimum(qdm(x0c.astype(np.float64), np.sort(crefs, 0), mu17c, alpha), scen_tot)
    scen_strat = scen_tot - scen_conv
    scen_tot, scen_conv, scen_strat = (f.astype(np.float32) for f in (scen_tot, scen_conv, scen_strat))

    out = f"{V17}/scenario_3comp_{ymd}{cc}.npz"
    np.savez(out, scen_tot=scen_tot, scen_conv=scen_conv, scen_strat=scen_strat,
             x0t=x0t, x0c=x0c, x0s=x0s, lat=lat, lon=lon, alpha=alpha, n_ref=n)
    print(f"\n💾 {out}  (n_ref={n}, α={alpha:.5f})")
    print(f"   نطاق: كلي  mean={scen_tot.mean():.2f} max={scen_tot.max():.1f}"
          f" | حملي mean={scen_conv.mean():.2f} max={scen_conv.max():.1f}"
          f" | طبقي mean={scen_strat.mean():.2f} max={scen_strat.max():.1f}")

    print(f"\n📊 المحطات (دورة {ymd[6:]}/{cc}Z) — سيناريو (خام):")
    print("-" * 78)
    print(f"{'المحطة':<16} {'كلي':>6} {'حملي':>6} {'طبقي':>6} {'حصة حملي':>9}   خام (ك/ح/ط)")
    print("-" * 78)
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lat - la).argmin()); j = int(np.abs(lon - lo).argmin())
        vt, vc, vs = (float(f[i, j]) for f in (scen_tot, scen_conv, scen_strat))
        share = vc / vt if vt > 0.5 else float('nan')
        print(f"  {name:<16} {vt:>5.1f} {vc:>5.1f} {vs:>5.1f} {share:>8.0%}"
              f"   ({x0t[i,j]:>4.0f}/{x0c[i,j]:>4.0f}/{x0s[i,j]:>4.0f})")
    print("-" * 78)


if __name__ == "__main__":
    ymd = sys.argv[1] if len(sys.argv) > 1 else "20260914"
    cc = (sys.argv[2] if len(sys.argv) > 2 else "00").zfill(2)
    main(ymd, cc)
