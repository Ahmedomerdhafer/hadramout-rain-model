#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  بناء سيناريو ثلاثي المكوّنات (كلي/حملي/طبقي) لنافذة الـ24 ساعة القادمة
  build_3comp_24h.py YYYYMMDD CC   (مثال: python3 build_3comp_24h.py 20260914 06)
===============================================================================
  السلسلة:
    1) المشغّل: APCP + ACPCP التراكمية 0–24 ساعة من ملف f024 لأحدث دورة
       (رسالة "0-1 day acc fcst" — التنزيل عبر byte-range من الفهرس)
    2) عينة مرجعية 24 ساعة: نفس 24 دورة مرجعية المعتمدة (0–24 ساعة من f024)
    3) معايرة QDM (Cannon 2015) — الوصفة المُتحقَّق بها (فرق ≤0.17 ملم):
         q = clip((cnt+0.5)/(n+1), 1/(n+1), n/(n+1)) ، حد غاما = 0 حيث mu=0
    4) مناخ v17-HR1 مقيس إلى 24 ساعة:  mu17_24 = mu17_10day × r
       حيث r = نسبة (متوسط 0–24س / متوسط 0–240س) في العينة المرجعية نفسها،
       منعّمة 3×3 ومقصوصة [0,1] — نفس منهجية النسبة الحملية f_ref (82.6%)
    5) الحملي: mu17c_24 = mu17_24 × f_ref مع سقف ≤ الكلي؛ الطبقي = الكلي − الحملي
  ملاحظة: شكل غاما α = 0.4845 (من v17-HR1) مثبت للنافذتين — افتراض موثق.
===============================================================================
"""
import sys, os, re, glob
import numpy as np
import requests
import eccodes
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import gamma
from scipy.ndimage import uniform_filter

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CACHE = "gfs_cache/op"
IDX24 = f"{CACHE}/idx24"
V17 = "gfs_cache/v17replay"
LAT0, LAT1, LON0, LON1 = 11.0, 21.0, 44.0, 56.0
STATIONS = {
    'Mukalla': (49.12, 14.53), 'Ash Shihr': (49.60, 14.76),
    'Al-Dhabba Port': (49.50, 14.70), 'Shuhayr': (49.42, 14.65),
    'Wadi Huwayrah': (49.35, 14.82), 'Seiyun (Wadi)': (48.78, 15.93),
    'Tarim': (49.00, 16.05), 'Dawan (Wadi)': (48.45, 15.00),
    'Al-Ghaydah': (52.19, 16.21),
}


def parse_idx(text):
    """تحليل فهرس .idx — السطور بصيغتين: 'n:n:byte:d=..' أو 'n:byte:d=..'"""
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
        entries.append((byte, f[di + 1] if len(f) > di + 1 else '',
                        f[di + 2] if len(f) > di + 2 else '',
                        f[di + 3] if len(f) > di + 3 else ''))
    return entries


def crop_flip(vals):
    """قص النطاق من الشبكة العالمية (صفوف تنازلية العرض) ثم قلبها تصاعدياً"""
    r0 = int(round((90.0 - LAT1) / 0.25))   # 276 (lat 21)
    r1 = int(round((90.0 - LAT0) / 0.25))   # 316 (lat 11)
    c0 = int(round(LON0 / 0.25))            # 176 (lon 44)
    c1 = int(round(LON1 / 0.25))            # 224 (lon 56)
    return vals[r0:r1 + 1, c0:c1 + 1][::-1, :].astype(np.float32)


def qdm(x0, refs_sorted, mu, alpha):
    """QDM المُتحقَّق منه (يعيد إنتاج سلسلة الأيام العشرة بدقة ≤ 0.17 ملم)"""
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


def get_idx(ymd, cc, fhour):
    p = f"{IDX24}/gfs_{ymd}{cc}_f{fhour}.idx"
    if not os.path.exists(p):
        r = requests.get(f"{BASE}/gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{fhour}.idx",
                         timeout=120)
        r.raise_for_status()
        open(p, 'w').write(r.text)
    return p


def fetch_message(ymd, cc, fhour, var, idx_path):
    """تنزيل رسالة GRIB التراكمية 0–24 ساعة عبر byte-range"""
    url = f"{BASE}/gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{fhour}"
    entries = parse_idx(open(idx_path).read())
    hits = [(b, d) for (b, v, l, d) in entries
            if v == var and l == 'surface' and re.search(r'0-(24 hour|1 day) acc', d)]
    assert len(hits) == 1, f"{var}: توقعت رسالة واحدة، وجدت {len(hits)}"
    start = hits[0][0]
    later = [b for (b, _, _, _) in entries if b > start]
    end = (min(later) - 1) if later else int(
        requests.head(url, timeout=60).headers['Content-Length']) - 1
    blob = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=300).content
    tmp = f"{CACHE}/msg24_{ymd}{cc}_{var}.bin"
    open(tmp, 'wb').write(blob)
    with open(tmp, 'rb') as fh:
        g = eccodes.codes_grib_new_from_file(fh)
        short, step = eccodes.codes_get(g, 'shortName'), eccodes.codes_get(g, 'stepRange')
        vals = eccodes.codes_get_values(g).reshape(721, 1440)
        eccodes.codes_release(g)
    os.remove(tmp)
    assert short in ('tp', 'acpcp') and step == '0-24', f"{var}: {short}/{step}؟!"
    return vals


def fetch_cycle(lab):
    """رسالتا 0–24 ساعة لدورة مرجعية واحدة (مع إعادات محاولة)"""
    ymd, cc = lab[:8], lab[8:]
    for attempt in range(3):
        try:
            idx = get_idx(ymd, cc, "024")
            t = crop_flip(fetch_message(ymd, cc, "024", "APCP", idx))
            c = crop_flip(fetch_message(ymd, cc, "024", "ACPCP", idx))
            return lab, t, c
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  ↻ إعادة محاولة {lab}: {e}")


def main(ymd, cc):
    os.makedirs(IDX24, exist_ok=True)
    labels = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{V17}/v16ref/*.npy"))
    assert len(labels) == 24, f"توقعت 24 دورة مرجعية، وجدت {len(labels)}"

    # ===== العينة المرجعية 0–24 ساعة (تُخزَّن للاستخدام اللاحق) =====
    ref_npz = f"{V17}/ref24_sample.npz"
    if os.path.exists(ref_npz):
        d = np.load(ref_npz, allow_pickle=True)
        if list(d["labels"]) == labels:
            t24, c24 = d["t24"], d["c24"]
            print(f"✅ عينة مرجعية 0–24 ساعة محمّلة من {ref_npz}")
        else:
            os.remove(ref_npz)
    if not os.path.exists(ref_npz):
        print("📡 تنزيل العينة المرجعية 0–24 ساعة (24 دورة × رسالتين)...")
        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(fetch_cycle, labels))
        res.sort(key=lambda r: r[0])
        t24 = np.stack([r[1] for r in res])
        c24 = np.stack([r[2] for r in res])
        np.savez(ref_npz, t24=t24, c24=c24, labels=np.array(labels))
        print(f"✅ حُفظت العينة المرجعية 0–24 ساعة → {ref_npz}")
    assert t24.shape == (24, 41, 49), t24.shape

    # ===== المشغّل: أحدث دورة =====
    print(f"\n📡 المشغّل — دورة {ymd}/{cc}Z (رسالتا 0–24 ساعة):")
    idx = get_idx(ymd, cc, "024")
    x0t = crop_flip(fetch_message(ymd, cc, "024", "APCP", idx))
    x0c = crop_flip(fetch_message(ymd, cc, "024", "ACPCP", idx))
    if (x0c > x0t + 1e-6).any():
        print(f"  ⚠️ {(x0c > x0t + 1e-6).sum()} نقطة ACPCP>APCP — قُصّت")
        x0c = np.minimum(x0c, x0t)
    x0s = np.clip(x0t - x0c, 0, None).astype(np.float32)

    # ===== المعاملات + نسبة القياس الزمني r =====
    db = np.load(f"{V17}/scenario_20260912.npz")
    mu17 = db["mu17"].astype(np.float64)
    alpha = float(db["alpha"])
    f_ref = np.load(f"{V17}/scenario_conv_2026091306.npz")["f_ref"].astype(np.float64)
    ref10 = np.stack([np.load(p)[::-1, :] for p in
                      sorted(glob.glob(f"{V17}/v16ref/*.npy"))]).astype(np.float64)

    num = t24.astype(np.float64).sum(axis=0)
    den = ref10.sum(axis=0)
    r = np.where(den > 0.1, num / np.maximum(den, 1e-9), 0.0)
    r = np.clip(uniform_filter(r, size=3, mode='nearest'), 0.0, 1.0)
    mu17_24 = mu17 * r
    mu17c_24 = mu17_24 * f_ref
    m = mu17 > 0.3
    print(f"\n⚙️ نسبة القياس الزمني r (24س/240س مرجعية): متوسط {r[m].mean():.1%} "
          f"| نطاق [{r[m].min():.1%}, {r[m].max():.1%}] على نطاق mu17>0.3")

    # ===== QDM لنافذة 24 ساعة =====
    scen_tot = qdm(x0t.astype(np.float64), np.sort(t24.astype(np.float64), 0), mu17_24, alpha)
    scen_conv = np.minimum(qdm(x0c.astype(np.float64), np.sort(c24.astype(np.float64), 0),
                               mu17c_24, alpha), scen_tot)
    scen_strat = scen_tot - scen_conv
    scen_tot, scen_conv, scen_strat = (f.astype(np.float32) for f in (scen_tot, scen_conv, scen_strat))

    lat = np.round(np.arange(LAT0, LAT1 + 1e-9, 0.25), 2)
    lon = np.round(np.arange(LON0, LON1 + 1e-9, 0.25), 2)
    out = f"{V17}/scenario_3comp_24h_{ymd}{cc}.npz"
    np.savez(out, scen_tot=scen_tot, scen_conv=scen_conv, scen_strat=scen_strat,
             x0t=x0t, x0c=x0c, x0s=x0s, lat=lat, lon=lon,
             alpha=alpha, n_ref=24, r24=r, mu17_24=mu17_24, mu17c_24=mu17c_24)
    print(f"\n💾 {out}")
    print(f"   نطاق 24 ساعة: كلي mean={scen_tot.mean():.2f} max={scen_tot.max():.1f}"
          f" | حملي mean={scen_conv.mean():.2f} max={scen_conv.max():.1f}"
          f" | طبقي mean={scen_strat.mean():.2f} max={scen_strat.max():.1f}")
    print(f"   خام:          كلي max={x0t.max():.1f} | حملي max={x0c.max():.1f} | طبقي max={x0s.max():.1f}")

    # ===== تقرير المحطات + مقارنة مع سيناريو الأيام العشرة =====
    d10 = np.load(f"{V17}/scenario_3comp_2026091400.npz") if \
        os.path.exists(f"{V17}/scenario_3comp_2026091400.npz") else None
    print(f"\n📊 المحطات (دورة {ymd[6:]}/{cc}Z — نافذة 24 ساعة):")
    print("-" * 88)
    print(f"{'المحطة':<16} {'كلي':>6} {'حملي':>6} {'طبقي':>6} {'حصة حملي':>9}"
          f"   خام (ك/ح/ط)   {'كلي 10 أيام':>12}")
    print("-" * 88)
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lat - la).argmin()); j = int(np.abs(lon - lo).argmin())
        vt, vc, vs = (float(f[i, j]) for f in (scen_tot, scen_conv, scen_strat))
        share = vc / vt if vt > 0.5 else float('nan')
        extra = f"{float(d10['scen_tot'][i, j]):>12.1f}" if d10 is not None else ""
        print(f"  {name:<16} {vt:>5.1f} {vc:>5.1f} {vs:>5.1f} {share:>8.0%}"
              f"   ({x0t[i,j]:>4.0f}/{x0c[i,j]:>4.0f}/{x0s[i,j]:>4.0f}){extra}")
    print("-" * 88)


if __name__ == "__main__":
    ymd = sys.argv[1] if len(sys.argv) > 1 else "20260914"
    cc = (sys.argv[2] if len(sys.argv) > 2 else "06").zfill(2)
    main(ymd, cc)
