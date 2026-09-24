# -*- coding: utf-8 -*-
"""
===============================================================================
  بحر العرب وخليج عدن — التوقعات المدارية (المنخفضات والأعاصير) خلال 14 يوماً
===============================================================================
  الاستخدام: python3 arabian_sea_tropical_14d.py [YYYYMMDD] [CC]
  المصدر   : GFS v16.3 0.25° (AWS open data) — حتى f336 (اليوم 14)
  اللوحات  : (1) تراكم مطري 14 يوماً  (2) أقصى سرعة رياح 10م
             (3) الضغط الأدنى + مسارات المنخفضات المكتشفة تلقائياً
  تنويه    : منتج استرشادي من نموذج حتمي مفرد — ليس تحذيراً رسمياً
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

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CACHE = "gfs_cache/op"
V17 = "gfs_cache/v17replay"

# نطاق بحر العرب: من عدن/باب المندب غرباً إلى كيرالا شرقاً، ومن الساحل الإيراني
# شمالاً إلى شمال المحيط الهندي جنوباً
LON0, LON1, LAT0, LAT1 = 44.5, 77.5, -5.0, 26.0
EXTENT = [LON0, LON1, LAT0, LAT1]

# خطوات زمنية: 6-ساعية حتى اليوم 7.5 ثم 12-ساعية حتى اليوم 14
STEPS = list(range(0, 181, 6)) + list(range(192, 337, 12))

CITIES = [
    ("المكلا", 49.12, 14.53),
    ("حديبو (سقطرى)", 54.02, 12.55),
    ("صلالة", 54.09, 17.01),
    ("عدن", 45.02, 12.79),
    ("كوتشي (الهند)", 76.27, 9.97),
    ("مومباي", 72.86, 19.02),
]

# ---- الترقيم والألوان: مطر WeatherBELL (كما في بقية المنتجات) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from hadramout_gfs_10day_rain import _ar, _setup_arabic_font
from wxbell_legend import draw_wxbell_legend, WX_LEVELS, WX_COLORS, WX_OVER

# رياح (كم/س): 10 فئات + فئة ≥250
WIND_LEVELS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250]
WIND_COLORS = ["#C9E9F7", "#8FD3F2", "#5BB8E8", "#39A0DB", "#FFE066",
               "#FFC53D", "#F5941C", "#E85D10", "#D22C05", "#A30F0F"]
WIND_OVER = "#8E24AA"

# الضغط الأدنى (hPa): الأدنى = الأقوى
PRES_LEVELS = [980, 985, 990, 995, 1000, 1004, 1008]
PRES_COLORS = ["#4A148C", "#7B1FA2", "#AB47BC", "#CE93D8", "#E1BEE7",
               "#F3E5F5", "#FBEFF9"]
PRES_OVER = "#FDF8FD"


def _get(url, tries=3, headers=None):
    for k in range(tries):
        try:
            return requests.get(url, timeout=300, headers=headers)
        except Exception:
            if k == tries - 1:
                raise
            import time
            time.sleep(2 * (k + 1))


def fetch_crop(ymd, cc, fxx, var, lev, pat=None):
    """تنزيل رسالة GRIB واحدة (byte-range) وقصّها لنطاق بحر العرب"""
    tag = f"as14d_{ymd}{cc}_f{fxx:03d}_{var}"
    idx_path = os.path.join(CACHE, f"{tag}.idx")
    bin_path = os.path.join(CACHE, f"{tag}.bin")
    url = f"{BASE}/gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{fxx:03d}"
    if not os.path.exists(idx_path):
        r = _get(f"{url}.idx")
        r.raise_for_status()
        open(idx_path, "w").write(r.text)
    from build_3comp_scenario import parse_idx
    entries = parse_idx(open(idx_path).read())
    hits = [(b, d) for (b, v, l, d) in entries if v == var and l == lev
            and (pat is None or re.search(pat, d))]
    assert len(hits) == 1, f"{var}/{lev}@f{fxx}: {len(hits)} رسالة"
    start = hits[0][0]
    later = [b for (b, _, _, _) in entries if b > start]
    if not os.path.exists(bin_path):
        if later:
            end = min(later) - 1
        else:
            end = int(requests.head(url, timeout=60).headers["Content-Length"]) - 1
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
    return vals[r0:r1 + 1, c0:c1 + 1][::-1, :].astype(np.float32)  # تصاعدي


def ocean_mask():
    """قناع المحيط على شبكة النطاق (لاكتشاف المنخفضات فوق البحر فقط)"""
    import shapely
    from shapely import contains_xy
    from shapely.ops import unary_union
    land = unary_union(list(cfeature.LAND.with_scale("50m").geometries()))
    lats = np.arange(LAT0, LAT1 + 0.001, 0.25)
    lons = np.arange(LON0, LON1 + 0.001, 0.25)
    LON, LAT = np.meshgrid(lons, lats)
    return ~contains_xy(land, LON, LAT)          # True = محيط


def detect_tracks(pres_steps, lons, lats, sea):
    """كشف مسارات المنخفضات: أدنى ضغط محلي متتبَّع عبر الخطوات"""
    tracks = []                                    # كل مسار: نقاط/ضغوط/خطوات
    for (fhr, psl) in pres_steps:
        mn = ndimage.minimum_filter(psl, size=5, mode="nearest")
        cand = []
        for i in range(psl.shape[0]):
            for j in range(psl.shape[1]):
                if psl[i, j] == mn[i, j] and psl[i, j] <= 1005.5:
                    near_land = any(
                        (j * 0.25 + LON0 - t["pts"][-1][0]) ** 2 +
                        (i * 0.25 + LAT0 - t["pts"][-1][1]) ** 2 < 25
                        for t in tracks if t["open"])
                    if sea[i, j] or near_land:
                        cand.append((j, i, float(psl[i, j])))
        cand.sort(key=lambda c: c[2])
        dt = 6
        if tracks and tracks[-1].get("last_fhr") is not None:
            dt = fhr - tracks[-1]["last_fhr"]
        reach = 4.5 * dt / 6.0
        extended = set()
        for (j, i, p) in cand:
            lon, lat = j * 0.25 + LON0, i * 0.25 + LAT0
            best, bd = None, 9e9
            for t in tracks:
                if not t["open"]:
                    continue
                lx, ly = t["pts"][-1]
                d = (lon - lx) ** 2 + (lat - ly) ** 2
                if d < bd:
                    best, bd = t, d
            if best is not None and bd <= reach ** 2:
                if id(best) not in extended:      # نقطة واحدة لكل مسار في كل خطوة
                    best["pts"].append((lon, lat))
                    best["prs"].append(p)
                    best["fhrs"].append(fhr)
                    extended.add(id(best))
            elif sea[i, j]:
                t = {"pts": [(lon, lat)], "prs": [p],
                     "fhrs": [fhr], "open": True}
                tracks.append(t)
                extended.add(id(t))
        for t in tracks:
            t["open"] = t["fhrs"][-1] == fhr
            t["last_fhr"] = fhr
    keep = [t for t in tracks
            if len(t["pts"]) >= 4 and min(t["prs"]) <= 1004.5]
    keep.sort(key=lambda t: min(t["prs"]))
    return keep


def build(ymd, cc):
    out = os.path.join(V17, f"arabian_sea_14d_{ymd}{cc}.npz")
    if os.path.exists(out):
        print(f"✅ النموذج مبني مسبقاً: {out}")
        return out
    os.makedirs(CACHE, exist_ok=True)
    tasks = []
    for fxx in STEPS:
        for var, lev in (("PRMSL", "mean sea level"),
                         ("UGRD", "10 m above ground"),
                         ("VGRD", "10 m above ground")):
            tasks.append((fxx, var, lev, None))
    tasks.append((336, "APCP", "surface", "0-14 day"))

    data = {}
    done = [0]
    def work(t):
        fxx, var, lev, pat = t
        v = fetch_crop(ymd, cc, fxx, var, lev, pat)
        return (fxx, var, v)

    print(f"📡 تنزيل {len(tasks)} رسالة (ضغط + رياح 10م × {len(STEPS)} خطوة + مطر f336)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fu in as_completed(futs):
            fxx, var, v = fu.result()
            data[(fxx, var)] = v
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"   … {done[0]}/{len(tasks)}")

    lats = np.arange(LAT0, LAT1 + 0.001, 0.25)
    lons = np.arange(LON0, LON1 + 0.001, 0.25)
    rain14 = data[(336, "APCP")]
    vmax = np.zeros_like(rain14)
    pmin = np.full_like(rain14, 1100.0)
    pres_steps = []
    for fxx in STEPS:
        u = data[(fxx, "UGRD")]
        v = data[(fxx, "VGRD")]
        spd = np.sqrt(u ** 2 + v ** 2) * 3.6
        vmax = np.maximum(vmax, spd)
        psl = data[(fxx, "PRMSL")] / 100.0
        pmin = np.minimum(pmin, psl)
        pres_steps.append((fxx, psl))

    print("🌀 كشف مسارات المنخفضات فوق البحر...")
    sea = ocean_mask()
    tracks = detect_tracks(pres_steps, lons, lats, sea)
    print(f"   أنظمة مكتشفة (≥24 ساعة): {len(tracks)}")
    for k, t in enumerate(tracks):
        print(f"   S{k+1}: أدنى ضغط {min(t['prs']):.0f} hPa | "
              f"{len(t['pts'])} نقطة | f{t['fhrs'][0]}→f{t['fhrs'][-1]}")

    t0 = datetime.strptime(ymd + cc, "%Y%m%d%H")
    np.savez_compressed(
        out, lat=lats, lon=lons, rain14=rain14, vmax=vmax, pmin=pmin,
        pres3d=np.stack([p for (_, p) in pres_steps]).astype(np.float32),
        fhrs=np.array([f for (f, _) in pres_steps]),
        tracks_json=json.dumps(tracks), t0=np.datetime64(t0))
    # نظّف المخزن المؤقت (الرسائل الخام) حفاظاً على حصة القرص
    import glob
    for f in glob.glob(os.path.join(CACHE, "as14d_*")):
        os.remove(f)
    print(f"💾 {out}")
    return out


def plot(ymd, cc):
    src = os.path.join(V17, f"arabian_sea_14d_{ymd}{cc}.npz")
    d = np.load(src, allow_pickle=True)
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
            norm = BoundaryNorm(WX_LEVELS, cmap.N)
            cf = ax.contourf(lons, lats, f, levels=WX_LEVELS, cmap=cmap, norm=norm,
                             extend="max", zorder=2, transform=ccrs.PlateCarree())
            ax.text(0.985, 0.02,
                    f"max: {np.nanmax(rain14):.0f} mm | mean: {np.nanmean(rain14):.1f} mm",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.75, edgecolor="#bbb"))
        elif kind == "wind":
            f = np.where(vmax >= WIND_LEVELS[0], vmax, np.nan)
            cmap = ListedColormap(WIND_COLORS); cmap.set_over(WIND_OVER); cmap.set_bad(alpha=0)
            norm = BoundaryNorm(WIND_LEVELS, cmap.N)
            ax.contourf(lons, lats, f, levels=WIND_LEVELS, cmap=cmap, norm=norm,
                        extend="max", zorder=2, transform=ccrs.PlateCarree())
            ax.text(0.985, 0.02, f"peak: {vmax.max():.0f} km/h",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.75, edgecolor="#bbb"))
        else:
            cmap = ListedColormap(PRES_COLORS)
            cmap.set_over(PRES_OVER)
            cmap.set_under("#2E0A5E")
            norm = BoundaryNorm(PRES_LEVELS, cmap.N)
            ax.contourf(lons, lats, pmin, levels=PRES_LEVELS, cmap=cmap, norm=norm,
                        extend="both", zorder=2, transform=ccrs.PlateCarree())
            cs = ax.contour(lons, lats, pmin, levels=[984, 988, 992, 996, 1000, 1004],
                            colors="#37474F", linewidths=0.55,
                            transform=ccrs.PlateCarree(), zorder=4)
            ax.clabel(cs, fontsize=6, fmt="%d")
            ax.text(0.985, 0.02, f"min MSLP: {pmin.min():.0f} hPa",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.75, edgecolor="#bbb"))
            # مسارات المنخفضات المكتشفة
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

    # مدن مرجعية (نقاط + أسماء عبر الطريقة الموثوقة على مستوى الشكل)
    # نقاط مرجعية بلا أسماء (أُزيلت أسماء المناطق بطلب المستخدم — القيم في جدول CSV)
    for ax in axes:
        for (name, lo, la) in CITIES:
            ax.plot(lo, la, "o", ms=4.5, color="darkred", zorder=6,
                    transform=ccrs.PlateCarree())

    # صندوق النافذة فوق اللوحة الأولى
    axes[0].text(0.015, 0.985,
                 f"Window: next 14 days — {t0.strftime('%d %b %Y')} → "
                 f"{t1.strftime('%d %b %Y')} ({ymd}/{cc}Z)\n"
                 "GFS v16.3 0.25° deterministic — 6-hourly to day 7.5, 12-hourly after",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5,
                 color="#222", fontweight="bold", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           alpha=0.8, edgecolor="#bbb"))

    # المفاتيح المرقّمة تحت كل لوحة (نفس طراز الصورة المرجعية)
    fig.subplots_adjust(left=0.020, right=0.986, top=0.865, bottom=0.239, wspace=0.05)
    p1, p2, p3 = (a.get_position() for a in axes)
    draw_wxbell_legend(fig, x0=p1.x0, x1=p1.x1, y_bar=0.168, bar_h=0.020,
                       num_fs=6.6, unit_fs=8.0, arabic_font=arabic_fonts, ar=_ar)
    draw_wxbell_legend(fig, x0=p2.x0, x1=p2.x1, y_bar=0.168, bar_h=0.020,
                       levels=WIND_LEVELS, colors=WIND_COLORS, over_color=WIND_OVER,
                       unit_ar="كم/س", num_fs=7.5, unit_fs=8.0,
                       arabic_font=arabic_fonts, ar=_ar)
    draw_wxbell_legend(fig, x0=p3.x0, x1=p3.x1, y_bar=0.168, bar_h=0.020,
                       levels=PRES_LEVELS, colors=PRES_COLORS, over_color=PRES_OVER,
                       unit_ar="hPa", num_fs=7.5, unit_fs=8.0)

    # العنوان الرئيسي
    cx = (p1.x0 + p3.x1) / 2.0
    fig.suptitle(_ar("بحر العرب وخليج عدن — التوقعات المدارية (المنخفضات والأعاصير) خلال 14 يوماً القادمة\n"
                     "التراكم المطري وأقصى الرياح والضغط الأدنى مع مسارات المنخفضات المتوقعة — نمذجة NOAA GFS v16.3 | تصميم: أحمد عمر ظافر"),
                 fontsize=15, fontweight="bold", x=cx, ha="center",
                 fontfamily=arabic_fonts, y=0.995)

    fig.text(0.008, 0.004,
             "Arabian Sea & Gulf of Aden tropical outlook — next 14 days (0\u2013336 h) from a single deterministic GFS v16.3 0.25\u00b0 run  |  "
             "NOT an official tropical cyclone warning \u2014 refer to RSMC New Delhi (IMD) and national meteorological services\n"
             "Tracks = automatically detected 6-hourly sea-level-pressure minima persisting \u2265 24 h  |  "
             "Confidence decreases sharply beyond day 5: GFS 0.25\u00b0 underestimates cyclone intensity and may displace tracks\n"
             "Panels: (1) 14-day accumulated precipitation (APCP 0\u2013336 h, raw GFS \u2014 no scenario calibration)  "
             "(2) peak 10-m wind speed  (3) minimum mean-sea-level pressure with isobars & detected tracks (S\u2026 = system number, \u2605 = peak)\n"
             "Reference points: Mukalla \u00b7 Socotra (Hidaybu) \u00b7 Salalah \u00b7 Aden \u00b7 Kochi \u00b7 Mumbai  |  "
             "Boundaries: Natural Earth  |  Display grid 0.25\u00b0 (~25 km \u2014 raw model resolution)",
             fontsize=6.6, color="#555", ha="left", va="bottom")

    output = f"hadramout_v17_arabian_sea_14d_{ymd}_{cc}Z.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"✅ لوحة بحر العرب: {output}")

    # جدول المدن المرجعية + CSV
    print(f"\n📊 النقاط المرجعية (دورة {ymd}/{cc}Z — 14 يوماً):")
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
