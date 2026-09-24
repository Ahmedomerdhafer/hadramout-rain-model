# -*- coding: utf-8 -*-
"""
===============================================================================
  بحر العرب وخليج عدن — التوقعات المدارية خلال 14 يوماً — النموذج الذكي AI-GFS
===============================================================================
  الاستخدام : python3 arabian_sea_tropical_ai.py [YYYYMMDD] [CC]
  المصدر    : NOAA AIGFS v1.0 (GraphCast) — noaa-nws-graphcastgfs-pds (0.25°)
              حتى f384؛ نستخدم f336 (اليوم 14) — نفس نافذة نسخة GFS تماماً
  ملاحظة    : يستعير كشف المسارات والأقنعة المحيطية من سكربت نسخة GFS
              (arabian_sea_tropical_14d.py) — فقط مصدر البيانات يتبدل.
===============================================================================
"""
import os
import re
import sys
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import eccodes
from scipy import ndimage

from build_3comp_scenario import parse_idx
from arabian_sea_tropical_14d import (
    ocean_mask, detect_tracks, EXTENT, LON0, LON1, LAT0, LAT1, STEPS,
    WIND_LEVELS, WIND_COLORS, WIND_OVER, PRES_LEVELS, PRES_COLORS, PRES_OVER,
    CITIES,
)
from hadramout_gfs_10day_rain import _ar, _setup_arabic_font
from wxbell_legend import draw_wxbell_legend, WX_LEVELS, WX_COLORS, WX_OVER

BASE_AI = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
CACHE = "gfs_cache/aigfs"
V17 = "gfs_cache/v17replay"


def _get(url, tries=3, headers=None):
    for k in range(tries):
        try:
            return requests.get(url, timeout=300, headers=headers)
        except Exception:
            if k == tries - 1:
                raise
            import time
            time.sleep(2 * (k + 1))


def fetch_ai(ymd, cc, fxx, var, lev, pat=None):
    """رسالة GRIB واحدة من AIGFS عبر byte-range، مقصوصة لنطاق بحر العرب"""
    os.makedirs(CACHE, exist_ok=True)
    tag = f"ai14d_{ymd}{cc}_f{fxx:03d}_{var}"
    idx_path = os.path.join(CACHE, f"{tag}.idx")
    bin_path = os.path.join(CACHE, f"{tag}.bin")
    url = (f"{BASE_AI}/aigfs.{ymd}/{cc}/model/atmos/grib2/"
           f"aigfs.t{cc}z.sfc.f{fxx:03d}.grib2")
    if not os.path.exists(idx_path):
        r = _get(f"{url}.idx")
        r.raise_for_status()
        open(idx_path, "w").write(r.text)
    entries = parse_idx(open(idx_path).read())
    hits = [(b, d) for (b, v, l, d) in entries
            if v == var and l == lev and (pat is None or re.search(pat, d))]
    assert len(hits) == 1, f"{var}/{lev}@f{fxx}: {len(hits)} رسالة"
    start = hits[0][0]
    if not os.path.exists(bin_path):
        later = [b for (b, _, _, _) in entries if b > start]
        end = (min(later) - 1) if later else int(
            requests.head(url, timeout=60).headers["Content-Length"]) - 1
        blob = _get(url, headers={"Range": f"bytes={start}-{end}"}).content
        open(bin_path, "wb").write(blob)
    with open(bin_path, "rb") as fh:
        g = eccodes.codes_grib_new_from_file(fh)
        vals = eccodes.codes_get_values(g).reshape(721, 1440)
        eccodes.codes_release(g)
    r0 = int(round((90.0 - LAT1) / 0.25))
    r1 = int(round((90.0 - LAT0) / 0.25))
    c0 = int(round(LON0 / 0.25))
    c1 = int(round(LON1 / 0.25))
    return vals[r0:r1 + 1, c0:c1 + 1][::-1, :].astype(np.float32)


def build(ymd, cc):
    out = os.path.join(V17, f"arabian_sea_14d_ai_{ymd}{cc}.npz")
    if os.path.exists(out):
        print(f"✅ مبني مسبقاً: {out}")
        return out
    tasks = []
    for fxx in STEPS:
        for var, lev in (("PRMSL", "mean sea level"),
                         ("UGRD", "10 m above ground"),
                         ("VGRD", "10 m above ground")):
            tasks.append((fxx, var, lev, None))
    tasks.append((336, "APCP", "surface", "0-14 day"))

    data = {}
    print(f"🤖 AIGFS (GraphCast) — بحر العرب: تنزيل {len(tasks)} رسالة...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_ai, ymd, cc, *t): t for t in tasks}
        done = 0
        for fu in as_completed(futs):
            fxx, var, _, _ = futs[fu]
            data[(fxx, var)] = fu.result()
            done += 1
            if done % 20 == 0:
                print(f"   … {done}/{len(tasks)}")

    lats = np.arange(LAT0, LAT1 + 0.001, 0.25)
    lons = np.arange(LON0, LON1 + 0.001, 0.25)
    rain14 = data[(336, "APCP")]
    vmax = np.zeros_like(rain14)
    pmin = np.full_like(rain14, 1100.0)
    pres_steps = []
    for fxx in STEPS:
        u = data[(fxx, "UGRD")]
        v = data[(fxx, "VGRD")]
        vmax = np.maximum(vmax, np.sqrt(u ** 2 + v ** 2) * 3.6)
        psl = data[(fxx, "PRMSL")] / 100.0
        pmin = np.minimum(pmin, psl)
        pres_steps.append((fxx, psl))

    print("🌀 كشف مسارات المنخفضات فوق البحر...")
    sea = ocean_mask()
    tracks = detect_tracks(pres_steps, lons, lats, sea)
    print(f"   أنظمة مكتشفة (≥24 ساعة): {len(tracks)}")
    for k, t in enumerate(tracks):
        print(f"   S{k+1}: أدنى ضغط {min(t['prs']):.0f} hPa | {len(t['pts'])} نقطة | "
              f"f{t['fhrs'][0]}→f{t['fhrs'][-1]}")

    t0 = datetime.strptime(ymd + cc, "%Y%m%d%H")
    np.savez_compressed(out, lat=lats, lon=lons, rain14=rain14, vmax=vmax,
                        pmin=pmin, tracks_json=json.dumps(tracks),
                        t0=np.datetime64(t0))
    import glob
    for f in glob.glob(os.path.join(CACHE, "ai14d_*")):
        os.remove(f)
    print(f"💾 {out}")
    return out


def plot(ymd, cc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    d = np.load(os.path.join(V17, f"arabian_sea_14d_ai_{ymd}{cc}.npz"),
                allow_pickle=True)
    lats, lons = d["lat"], d["lon"]
    rain14, vmax, pmin = d["rain14"], d["vmax"], d["pmin"]
    tracks = json.loads(str(d["tracks_json"]))
    t0 = datetime.strptime(ymd + cc, "%Y%m%d%H")
    t1 = t0 + timedelta(hours=336)

    arabic_fonts = _setup_arabic_font()
    fig = plt.figure(figsize=(24, 11.5), dpi=150)
    panels = [
        ("rain", "التراكم المطري خلال 14 يوماً", "14-Day Accumulated Precipitation"),
        ("wind", "أقصى سرعة رياح (10 م)", "Peak 10-m Wind Speed (14 days)"),
        ("pres", "الضغط الأدنى + مسارات المنخفضات", "Minimum MSLP & Detected Tracks"),
    ]
    axes = []
    for k, (kind, tar, ten) in enumerate(panels):
        ax = plt.subplot(1, 3, k + 1, projection=ccrs.PlateCarree())
        axes.append(ax)
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d4e4f0", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f2e8", zorder=0)
        ax.coastlines("50m", linewidth=0.7, color="#515151", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6,
                       edgecolor="#8b8b8b", zorder=3)
        if kind == "rain":
            f = np.where(rain14 >= WX_LEVELS[0], rain14, np.nan)
            cmap = ListedColormap(WX_COLORS); cmap.set_over(WX_OVER); cmap.set_bad(alpha=0)
            ax.contourf(lons, lats, f, levels=WX_LEVELS, cmap=cmap,
                        norm=BoundaryNorm(WX_LEVELS, cmap.N), extend="max",
                        zorder=2, transform=ccrs.PlateCarree())
            ax.text(0.985, 0.02,
                    f"max: {np.nanmax(rain14):.0f} mm | mean: {np.nanmean(rain14):.1f} mm",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.75, edgecolor="#bbb"))
        elif kind == "wind":
            f = np.where(vmax >= WIND_LEVELS[0], vmax, np.nan)
            cmap = ListedColormap(WIND_COLORS); cmap.set_over(WIND_OVER); cmap.set_bad(alpha=0)
            ax.contourf(lons, lats, f, levels=WIND_LEVELS, cmap=cmap,
                        norm=BoundaryNorm(WIND_LEVELS, cmap.N), extend="max",
                        zorder=2, transform=ccrs.PlateCarree())
            ax.text(0.985, 0.02, f"peak: {vmax.max():.0f} km/h",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.75, edgecolor="#bbb"))
        else:
            cmap = ListedColormap(PRES_COLORS)
            cmap.set_over(PRES_COLORS[-1]); cmap.set_under("#2E0A5E")
            ax.contourf(lons, lats, pmin, levels=PRES_LEVELS, cmap=cmap,
                        norm=BoundaryNorm(PRES_LEVELS, cmap.N), extend="both",
                        zorder=2, transform=ccrs.PlateCarree())
            cs = ax.contour(lons, lats, pmin, levels=[984, 988, 992, 996, 1000, 1004],
                            colors="#37474F", linewidths=0.55,
                            transform=ccrs.PlateCarree(), zorder=4)
            ax.clabel(cs, fontsize=6, fmt="%d")
            ax.text(0.985, 0.02, f"min MSLP: {pmin.min():.0f} hPa",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.75, edgecolor="#bbb"))
            for si, tr in enumerate(tracks[:4]):
                xs = [p[0] for p in tr["pts"]]
                ys = [p[1] for p in tr["pts"]]
                ax.plot(xs, ys, "-", color="#1A237E", lw=2, zorder=7,
                        transform=ccrs.PlateCarree())
                ax.plot(xs, ys, "o", color="#1A237E", ms=3.5, zorder=7,
                        transform=ccrs.PlateCarree())
                ipk = int(np.argmin(tr["prs"]))
                ax.plot(xs[ipk], ys[ipk], "*", ms=15, color="#FFD700",
                        markeredgecolor="#1A237E", zorder=8,
                        transform=ccrs.PlateCarree())
                pk_t = t0 + timedelta(hours=tr["fhrs"][ipk])
                ax.text(xs[ipk], ys[ipk] + 0.8,
                        f"S{si+1}: {min(tr['prs']):.0f} hPa\n{pk_t.strftime('%d/%m')}",
                        fontsize=7.5, fontweight="bold", color="#1A237E",
                        ha="center", va="bottom", zorder=8,
                        transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.35,
                          color="gray", alpha=0.4, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (k == 0)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 7, "color": "#333"}
        gl.ylabel_style = {"size": 7, "color": "#333"}
        ax.set_title(f"{ten}\n{_ar(tar)}", fontsize=10.5, fontweight="bold",
                     loc="center", pad=8, fontfamily=arabic_fonts)

    # نقاط مرجعية بلا أسماء (أُزيلت أسماء المناطق بطلب المستخدم — القيم في جدول CSV)
    for ax in axes:
        for (name, lo, la) in CITIES:
            ax.plot(lo, la, "o", ms=4.5, color="darkred", zorder=6,
                    transform=ccrs.PlateCarree())

    axes[0].text(0.015, 0.985,
                 f"Window: next 14 days — {t0.strftime('%d %b %Y')} → "
                 f"{t1.strftime('%d %b %Y')} ({ymd}/{cc}Z)\n"
                 "AI-GFS = NOAA AIGFS v1.0 (GraphCast) 0.25\u00b0 — 6-hourly to day 7.5, 12-hourly after",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5,
                 color="#222", fontweight="bold", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           alpha=0.8, edgecolor="#bbb"))

    fig.subplots_adjust(left=0.020, right=0.986, top=0.865, bottom=0.239, wspace=0.05)
    p1, p2, p3 = (a.get_position() for a in axes)
    draw_wxbell_legend(fig, x0=p1.x0, x1=p1.x1, y_bar=0.168, bar_h=0.020,
                       num_fs=6.6, unit_fs=8.0, arabic_font=arabic_fonts, ar=_ar)
    draw_wxbell_legend(fig, x0=p2.x0, x1=p2.x1, y_bar=0.168, bar_h=0.020,
                       levels=WIND_LEVELS, colors=WIND_COLORS, over_color=WIND_OVER,
                       unit_ar="كم/س", num_fs=7.5, unit_fs=8.0,
                       arabic_font=arabic_fonts, ar=_ar)
    draw_wxbell_legend(fig, x0=p3.x0, x1=p3.x1, y_bar=0.168, bar_h=0.020,
                       levels=PRES_LEVELS, colors=PRES_COLORS, over_color=PRES_COLORS[-1],
                       unit_ar="hPa", num_fs=7.5, unit_fs=8.0)

    cx = (p1.x0 + p3.x1) / 2.0
    fig.suptitle(_ar("بحر العرب وخليج عدن — التوقعات المدارية خلال 14 يوماً القادمة — النموذج الذكي NOAA AI-GFS (GraphCast)\n"
                     "التراكم المطري وأقصى الرياح والضغط الأدنى مع مسارات المنخفضات المتوقعة | تصميم: أحمد عمر ظافر"),
                 fontsize=15, fontweight="bold", x=cx, ha="center",
                 fontfamily=arabic_fonts, y=0.995)

    fig.text(0.008, 0.004,
             "Arabian Sea & Gulf of Aden tropical outlook — next 14 days (0\u2013336 h) from NOAA AIGFS v1.0 (GraphCast, DeepMind) 0.25\u00b0 — AI weather model initialized from GFS  |  "
             "NOT an official tropical cyclone warning \u2014 refer to RSMC New Delhi (IMD) and national meteorological services\n"
             "Tracks = automatically detected 6-hourly sea-level-pressure minima persisting \u2265 24 h  |  "
             "AI models tend to smooth intensity: deep lows may be stronger than depicted; confidence decreases sharply beyond day 5\n"
             "Panels: (1) 14-day accumulated precipitation (APCP 0\u2013336 h)  (2) peak 10-m wind speed  "
             "(3) minimum mean-sea-level pressure with isobars & detected tracks (S\u2026 = system, \u2605 = peak)  |  "
             "Reference points: Mukalla \u00b7 Socotra \u00b7 Salalah \u00b7 Aden \u00b7 Kochi \u00b7 Mumbai  |  Display grid 0.25\u00b0 (~25 km)",
             fontsize=6.6, color="#555", ha="left", va="bottom")

    output = f"hadramout_ai_arabian_sea_14d_{ymd}_{cc}Z.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"✅ لوحة بحر العرب (AI-GFS): {output}")

    print(f"\n📊 النقاط المرجعية (AI-GFS — دورة {ymd}/{cc}Z — 14 يوماً):")
    print("-" * 66)
    print(f"{'الموقع':<18} {'مطر 14ي':>9} {'أقصى رياح':>10} {'ضغط أدنى':>9}")
    print("-" * 66)
    rows = []
    for (name, lo, la) in CITIES:
        i = int(np.abs(lats - la).argmin())
        j = int(np.abs(lons - lo).argmin())
        r, w, p = float(rain14[i, j]), float(vmax[i, j]), float(pmin[i, j])
        rows.append((name, lo, la, r, w, p))
        print(f"  {name:<18} {r:>8.1f} {w:>9.0f} {p:>8.0f}")
    print("-" * 66)
    with open(output.replace(".png", "_cities.csv"), "w", encoding="utf-8") as f:
        f.write("city,longitude,latitude,rain_14d_mm,peak_wind_kmh,min_mslp_hpa\n")
        for (name, lo, la, r, w, p) in rows:
            f.write(f"{name},{lo:.2f},{la:.2f},{r:.1f},{w:.0f},{p:.0f}\n")
    print("✅ جدول المدن حُفظ")
    if tracks:
        print(f"\n🌀 ملخص الأنظمة المكتشفة ({len(tracks)}):")
        for si, tr in enumerate(tracks):
            ipk = int(np.argmin(tr["prs"]))
            pk_t = t0 + timedelta(hours=tr["fhrs"][ipk])
            st = t0 + timedelta(hours=tr["fhrs"][0])
            print(f"   S{si+1}: {min(tr['prs']):.0f} hPa عند {pk_t.strftime('%d/%m %H')}UTC "
                  f"(بدء {st.strftime('%d/%m')}, {len(tr['pts'])} نقطة)")


if __name__ == "__main__":
    ymd = sys.argv[1] if len(sys.argv) > 1 else "20260919"
    cc = (sys.argv[2] if len(sys.argv) > 2 else "00").zfill(2)
    build(ymd, cc)
    plot(ymd, cc)
