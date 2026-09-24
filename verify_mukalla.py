#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  تحقق أمام الرصد الأرضي — مطار الريان (OYRN، محطة المكلا)
  verify_mukalla.py [scenario_3comp_24h_*.npz]
===============================================================================
  يجلب بلاغات مطار الريان الساعية (METAR عبر أرشيف IEM) ويقارنها بـ:
    1) نافذة المنتج الحالي (المرصود حتى الآن مقابل قيمة المكلا)
    2) العينة المرجعية GFS (آخر 24 دورة) — معدل الإنذارات الكاذبة
  الاستخدام: python3 verify_mukalla.py   (يلتقط أحدث منتج 24 ساعة تلقائياً)
===============================================================================
"""
import csv
import glob
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

STATION_LAT, STATION_LON = 14.53, 49.12      # المكلا (خلية المنتج)
GAUGE = "OYRN"                               # مطار الريان (14.66N, 49.37E)


def fetch_obs(days=16):
    ets = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    sts = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%MZ")
    url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={GAUGE}"
           f"&data=p01i&sts={sts}&ets={ets}&tz=UTC&format=onlycomma"
           "&missing=M&trace=T&latlon=no&elev=no")
    import time as _time
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            break
        except Exception:
            if attempt == 3:
                raise
            _time.sleep(10 * (attempt + 1))
    obs = {}
    for row in csv.DictReader(r.text.splitlines()):
        if row["p01i"] in ("M", ""):
            continue
        t = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        obs[t] = 0.005 if row["p01i"] == "T" else float(row["p01i"]) * 25.4
    return obs


def window_obs(obs, t0, hours=24):
    have = sum(1 for k in range(1, hours + 1) if (t0 + timedelta(hours=k)) in obs)
    tot = sum(obs.get(t0 + timedelta(hours=k), 0.0) for k in range(1, hours + 1))
    return tot, have


def main(src=None):
    obs = fetch_obs()
    print(f"🌧️ رصد {GAUGE} (مطار الريان): {len(obs)} بلاغ ساعي — "
          f"مجموع الفترة: {sum(obs.values()):.1f} ملم")

    # ===== 1) المنتج الحالي =====
    if src is None:
        cands = sorted(glob.glob("gfs_cache/v17replay/scenario_3comp_24h_2*.npz"))
        assert cands, "لا يوجد منتج 24 ساعة"
        src = cands[-1]
    d = np.load(src)
    import re as _re
    m = _re.search(r'scenario_3comp_24h_(\d{10})\.npz', src)
    assert m, f"اسم ملف بلا دورة: {src}"
    t0 = datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=24)
    lat, lon = d["lat"], d["lon"]
    i = int(np.abs(lat - STATION_LAT).argmin())
    j = int(np.abs(lon - STATION_LON).argmin())
    elapsed = min((datetime.now(timezone.utc) - t0).total_seconds() / 3600, 24)
    tot, have = window_obs(obs, t0, hours=int(max(elapsed, 0)))
    print(f"\n📊 المنتج الحالي: {src.split('/')[-1]}")
    print(f"   نافذة {t0:%d %b %H:%M} → {t1:%d %b %H:%M} UTC")
    print(f"   المكلا — سيناريو: {float(d['scen_tot'][i, j]):.1f} ملم | خام: {float(d['x0t'][i, j]):.1f} ملم")
    if have > 0:
        print(f"   المرصود حتى الآن ({have} ساعة من أصل {int(elapsed)}): {tot:.1f} ملم")

    # ===== 2) تحقق العينة المرجعية GFS =====
    ref = np.load("gfs_cache/v17replay/ref24_sample.npz", allow_pickle=True)
    labels = [str(x) for x in ref["labels"]]
    t24 = ref["t24"]
    ri = int(np.abs(lat - STATION_LAT).argmin())
    rj = int(np.abs(lon - STATION_LON).argmin())
    hits, fa, dry_ok, n = 0, 0, 0, 0
    for k, lab in enumerate(labels):
        lt0 = datetime.strptime(lab, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        o, have = window_obs(obs, lt0)
        if have < 20:
            continue
        n += 1
        f = float(t24[k, ri, rj])
        if f >= 0.5 and o >= 0.5:
            hits += 1
        elif f >= 0.5:
            fa += 1
        elif f < 0.5 and o < 0.5:
            dry_ok += 1
    if n:
        print(f"\n🧪 تحقق العينة المرجعية ({n} نافذة):")
        print(f"   إنذارات صحيحة (≥0.5 ملم): {hits} | كاذبة: {fa} | جفاف صحيح: {dry_ok}")
        print(f"   معدل الإنذارات الكاذبة: {fa / max(hits + fa, 1):.0%}")
        print("   ⚠️ انحياز رطب إن كان المرصود صفراً دائماً مع تنبؤات متكررة")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
