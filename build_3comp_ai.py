# -*- coding: utf-8 -*-
"""
===============================================================================
  سيناريو v17-HR1 فوق النموذج الذكي AI-GFS (AIGFS v1.0 / GraphCast — NOAA EAGLE)
===============================================================================
  الاستخدام : python3 build_3comp_ai.py [YYYYMMDD] [CC]
  المصدر    : noaa-nws-graphcastgfs-pds — aigfs.tCCz.sfc (0.25°)
  المنهج    : نفس سلسلة GFS حرفياً — QDM (Cannon 2015) على نفس العينة
              المرجعية (24 دورة GFS v16.3) ونفس مناخ v17-HR1 (μ17، α=0.48447)
              ونفس نسبة القياس الزمني r (24س/240س) — فقط الحقل الخام يتبدل.
  المخرجات  : scenario_3comp_ai_<YYYYMMDDCC>.npz        (10 أيام)
              scenario_3comp_24h_ai_<YYYYMMDDCC>.npz     (24 ساعة)
===============================================================================
"""
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import eccodes

from build_3comp_scenario import (qdm, crop_flip, parse_idx,
                                  LAT0, LAT1, LON0, LON1)

BASE_AI = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
CACHE = "gfs_cache/aigfs"
V17 = "gfs_cache/v17replay"

STATIONS = {
    'Mukalla': (49.12, 14.53), 'Ash Shihr': (49.60, 14.76),
    'Al-Dhabba Port': (49.50, 14.70), 'Shuhayr': (49.42, 14.65),
    'Wadi Huwayrah': (49.35, 14.82), 'Seiyun (Wadi)': (48.78, 15.93),
    'Tarim': (49.00, 16.05), 'Dawan (Wadi)': (48.45, 15.00),
    'Al-Ghaydah': (52.19, 16.21),
}


def fetch_ai(ymd, cc, fxx, var, pat):
    """تنزيل رسالة GRIB واحدة من AIGFS عبر byte-range ثم قصّ نطاق حضرموت"""
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{ymd}{cc}_f{fxx:03d}_{var}"
    idx_path = os.path.join(CACHE, f"{tag}.idx")
    bin_path = os.path.join(CACHE, f"{tag}.bin")
    url = (f"{BASE_AI}/aigfs.{ymd}/{cc}/model/atmos/grib2/"
           f"aigfs.t{cc}z.sfc.f{fxx:03d}.grib2")
    if not os.path.exists(idx_path):
        r = requests.get(f"{url}.idx", timeout=120)
        r.raise_for_status()
        open(idx_path, "w").write(r.text)
    entries = parse_idx(open(idx_path).read())
    hits = [(b, d) for (b, v, l, d) in entries
            if v == var and l == "surface" and re.search(pat, d)]
    assert len(hits) == 1, f"{var}@f{fxx:03d} ({pat}): {len(hits)} رسالة: {hits}"
    start = hits[0][0]
    if not os.path.exists(bin_path):
        later = [b for (b, _, _, _) in entries if b > start]
        end = (min(later) - 1) if later else int(
            requests.head(url, timeout=60).headers["Content-Length"]) - 1
        blob = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                            timeout=300).content
        open(bin_path, "wb").write(blob)
    with open(bin_path, "rb") as fh:
        g = eccodes.codes_grib_new_from_file(fh)
        vals = eccodes.codes_get_values(g)
        eccodes.codes_release(g)
    return crop_flip(vals.reshape(721, 1440))


def main(ymd, cc):
    # ===== 1) التنزيل: تراكمات AIGPS الكلية فقط =====
    # (حقل ACPCP في AIGFS صوري = ثابت صفري عالمياً — تحقق مباشر، لذا تُشتق
    #  الحصة الحملية من f_ref المرجعية لمناخ v17 كما في منهج replay)
    jobs = [(240, "APCP", r"0-10 day acc"), (24, "APCP", r"0-1 day acc")]
    print(f"🤖 AIGFS v1.0 (GraphCast) — دورة {ymd}/{cc}Z: تنزيل {len(jobs)} رسالة...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda j: fetch_ai(ymd, cc, *j), jobs))

    x0t10 = res[0].astype(np.float64)
    x0t24 = res[1].astype(np.float64)

    # ===== 2) نفس مكوّنات منهج v17 (مناخ + مراجع GFS) =====
    db = np.load(f"{V17}/scenario_20260912.npz")
    mu17 = db["mu17"].astype(np.float64)
    alpha = float(db["alpha"])
    mu17c = np.load(f"{V17}/scenario_conv_2026091306.npz")["mu17_conv"].astype(np.float64)
    f_ref = np.load(f"{V17}/scenario_conv_2026091306.npz")["f_ref"].astype(np.float64)
    refs = np.stack([np.load(p)[::-1, :] for p in sorted(
        glob.glob(f"{V17}/v16ref/*.npy"))]).astype(np.float64)
    crefs = np.stack([np.load(p)[::-1, :] for p in sorted(
        glob.glob(f"{V17}/convref/*.npy"))]).astype(np.float64)
    dref = np.load(f"{V17}/ref24_sample.npz", allow_pickle=True)
    t24, c24 = dref["t24"].astype(np.float64), dref["c24"].astype(np.float64)

    # نسبة القياس الزمني r — نفس حقل r24 المخزن من سلسلة GFS
    from scipy.ndimage import uniform_filter
    ref10 = refs
    num = t24.sum(axis=0)
    den = ref10.sum(axis=0)
    r = np.where(den > 0.1, num / np.maximum(den, 1e-9), 0.0)
    r = np.clip(uniform_filter(r, size=3, mode="nearest"), 0.0, 1.0)
    mu17_24 = mu17 * r
    mu17c_24 = mu17_24 * f_ref

    # ===== 3) QDM — نفس الوصفة، والحصة الحملية من f_ref (لا حقل حملي في AIGFS) =====
    scen_tot10 = qdm(x0t10, np.sort(refs, 0), mu17, alpha)
    scen_conv10 = np.minimum(scen_tot10 * f_ref, scen_tot10)
    scen_strat10 = scen_tot10 - scen_conv10
    scen_tot24 = qdm(x0t24, np.sort(t24, 0), mu17_24, alpha)
    scen_conv24 = np.minimum(scen_tot24 * f_ref, scen_tot24)
    scen_strat24 = scen_tot24 - scen_conv24
    x0c10 = np.minimum(x0t10 * f_ref, x0t10).astype(np.float32)
    x0c24 = np.minimum(x0t24 * f_ref, x0t24).astype(np.float32)
    x0s10 = (x0t10 - x0c10).astype(np.float32)
    x0s24 = (x0t24 - x0c24).astype(np.float32)

    lat = np.round(np.arange(LAT0, LAT1 + 1e-9, 0.25), 2)   # نفس نطاق سلسلة v17
    lon = np.round(np.arange(LON0, LON1 + 1e-9, 0.25), 2)
    assert x0t10.shape == (len(lat), len(lon)), (x0t10.shape, len(lat), len(lon))
    for tag, st, sc, ss, xt, xc, xs in [
        ("", scen_tot10, scen_conv10, scen_strat10, x0t10, x0c10, x0s10),
        ("_24h", scen_tot24, scen_conv24, scen_strat24, x0t24, x0c24, x0s24),
    ]:
        out = f"{V17}/scenario_3comp{tag}_ai_{ymd}{cc}.npz"
        np.savez(out,
                 scen_tot=st.astype(np.float32), scen_conv=sc.astype(np.float32),
                 scen_strat=ss.astype(np.float32),
                 x0t=xt.astype(np.float32), x0c=xc.astype(np.float32),
                 x0s=xs.astype(np.float32), lat=lat, lon=lon, alpha=alpha, n_ref=24)
        print(f"💾 {out}")
        print(f"   كلي mean={st.mean():.2f} max={st.max():.1f} | "
              f"حملي mean={sc.mean():.2f} max={sc.max():.1f} | "
              f"طبقي mean={ss.mean():.2f} max={ss.max():.1f} | خام max={xt.max():.1f}")

    # ===== 4) تقرير المحطات مع مقارنة أمام GFS =====
    gfs10 = np.load(f"{V17}/scenario_3comp_{ymd}{cc}.npz") if os.path.exists(
        f"{V17}/scenario_3comp_{ymd}{cc}.npz") else None
    gfs24 = np.load(f"{V17}/scenario_3comp_24h_{ymd}{cc}.npz") if os.path.exists(
        f"{V17}/scenario_3comp_24h_{ymd}{cc}.npz") else None
    print(f"\n📊 المحطات — AI-GFS (GraphCast) مُعالَج بمنهج v17 — دورة {ymd[6:]}/{cc}Z:")
    print("-" * 84)
    print(f"{'المحطة':<16} {'AI-24س':>7} {'GFS-24س':>8} {'AI-10ي':>7} {'GFS-10ي':>8}"
          f"   {'AI خام (24س/10ي)':>18}")
    print("-" * 84)
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lat - la).argmin()); j = int(np.abs(lon - lo).argmin())
        a24 = float(scen_tot24[i, j]); a10 = float(scen_tot10[i, j])
        g24 = float(gfs24["scen_tot"][i, j]) if gfs24 is not None else float("nan")
        g10 = float(gfs10["scen_tot"][i, j]) if gfs10 is not None else float("nan")
        print(f"  {name:<16} {a24:>6.1f} {g24:>7.1f} {a10:>6.1f} {g10:>7.1f}"
              f"   ({x0t24[i,j]:>5.1f} / {x0t10[i,j]:>5.1f})")
    print("-" * 84)


if __name__ == "__main__":
    ymd = sys.argv[1] if len(sys.argv) > 1 else "20260919"
    cc = (sys.argv[2] if len(sys.argv) > 2 else "00").zfill(2)
    main(ymd, cc)
