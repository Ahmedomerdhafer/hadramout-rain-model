# -*- coding: utf-8 -*-
"""
===============================================================================
  خريطة مفردة: إجمالي الأمطار خلال الـ24 ساعة القادمة — محافظة حضرموت
  نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1
===============================================================================
  الاستخدام: python3 hadramout_v17_total_24h_map.py [YYYYMMDD] [CC]
  المصدر   : gfs_cache/v17replay/scenario_3comp_24h_<YYYYMMDDCC>.npz
===============================================================================
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from hadramout_gfs_10day_rain import (
    _ar, _setup_arabic_font, _add_yemen_admin1, _add_hadramout_districts,
    _interp_to_fine_grid, MAP_EXTENT, RAIN_MIN_MM, DISPLAY_RES_DEG,
)
from district_labels import add_district_labels

# دقة عرض أدق بطلب المستخدم: ~3 كم (كانت ~9 كم) — استيفاء من بيانات GFS 0.25°
DISPLAY_RES_DEG = 3.0 / 111.32

from wxbell_legend import draw_wxbell_legend, WX_LEVELS, WX_COLORS, WX_OVER
LEVELS = WX_LEVELS          # مفتاح WeatherBELL المرقّم — بطلب المستخدم
RAIN_COLORS = WX_COLORS
STATIONS = {
    'Mukalla': (49.12, 14.53), 'Ash Shihr': (49.60, 14.76),
    'Al-Dhabba Port': (49.50, 14.70), 'Shuhayr': (49.42, 14.65),
    'Wadi Huwayrah': (49.35, 14.82), 'Seiyun (Wadi)': (48.78, 15.93),
    'Tarim': (49.00, 16.05), 'Dawan (Wadi)': (48.45, 15.00),
    'Al-Ghaydah': (52.19, 16.21),
}

# ===== عامل العرض =====
# القيم الكاملة (أُلغيت النسخة المتحفظة ×0.5 بطلب المستخدم).
# يبقى التحقق الدائم أمام رصد مطار الريان (OYRN) عبر verify_mukalla.py —
# وهو يوثق انحيازاً رطباً لـ GFS في الأمطار الخفيفة على هذا الساحل.
CONSERVATIVE_FACTOR = 1.0


def main(ymd, cc):
    src = f"gfs_cache/v17replay/scenario_3comp_24h_{ymd}{cc}.npz"
    d = np.load(src)
    lats, lons = d["lat"], d["lon"]
    field = d["scen_tot"].astype(np.float64) * CONSERVATIVE_FACTOR
    x0t = d["x0t"]

    arabic_fonts = _setup_arabic_font()
    fig = plt.figure(figsize=(10.1, 13.0), dpi=150)   # نسبة تطابق البيانات → لا تقليص بـaspect
    ax = fig.add_axes([0.022, 0.205, 0.974, 0.722], projection=ccrs.PlateCarree())

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d4e4f0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f2e8", zorder=0)
    ax.coastlines("50m", linewidth=0.7, color="#515151", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8,
                   edgecolor="#8b8b8b", zorder=3)
    try:
        _add_yemen_admin1(ax)
    except Exception:
        pass
    try:
        _add_hadramout_districts(ax)
    except Exception:
        pass

    f_fine, lats_f, lons_f = _interp_to_fine_grid(field, lats, lons, DISPLAY_RES_DEG)
    f_plot = np.where(f_fine >= LEVELS[0], f_fine, np.nan)   # عتبة الفئة الأولى للمفتاح الجديد
    cmap = ListedColormap(RAIN_COLORS)
    cmap.set_over(WX_OVER)
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(LEVELS, cmap.N)
    cf = ax.contourf(lons_f, lats_f, f_plot, levels=LEVELS, cmap=cmap, norm=norm,
                     extend="max", alpha=1.0, zorder=2, transform=ccrs.PlateCarree())

    for st_lon, st_lat in STATIONS.values():
        ax.plot(st_lon, st_lat, marker='o', markersize=5, color='darkred',
                transform=ccrs.PlateCarree(), zorder=6)

    mlat = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    mlon = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    sub = field[np.ix_(mlat, mlon)]
    vmax = float(sub.max())
    imax = np.unravel_index(np.argmax(sub), sub.shape)
    vlat = float(lats[mlat][imax[0]]); vlon = float(lons[mlon][imax[1]])
    ax.text(0.985, 0.02,
            f"max: {vmax:.0f} mm ({vlat:.2f}N, {vlon:.2f}E)\nmean: {sub.mean():.1f} mm",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            color='#222222', zorder=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75,
                      edgecolor='#bbbbbb'))

    dd, mm = ymd[6:], ymd[4:6]
    dd1 = f"{int(dd)+1:02d}" if int(dd) < 30 else dd  # للعرض فقط
    ax.text(0.015, 0.985,
            f"Window: next 24 h — {mm} Sep {dd} {cc}Z → {dd1} Sep {cc}Z\n"
            f"Cycle: {ymd}/{cc}Z",
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
            color='#222222', fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.75, edgecolor='#bbbbbb'))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.45, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7.5, 'color': '#333333'}
    gl.ylabel_style = {'size': 7.5, 'color': '#333333'}

    # أسماء المديريات — بعد اكتمال بيانات المحاور (تحتاج transData نهائياً)
    n_lbl = add_district_labels(ax, _ar, arabic_fonts, fontsize=8.0, extent=MAP_EXTENT)
    print(f"🏷️ أسماء المديريات المرسومة: {n_lbl}")

    # مفتاح أفقي بطراز WeatherBELL: مربعات مرقّمة أسفل الخريطة (مثل الصورة المرفقة)
    draw_wxbell_legend(fig, x0=0.025, x1=0.975, y_bar=0.138, bar_h=0.020,
                       num_fs=7.6, unit_fs=8.6, arabic_font=arabic_fonts, ar=_ar)

    # العنوان موسّط فوق منطقة الخريطة (وليس مركز الشكل — الشريط الجانبي يزيحه)
    p = ax.get_position()
    cx = (p.x0 + p.x1) / 2.0
    fig.suptitle(_ar("محافظة حضرموت — إجمالي الأمطار خلال الـ24 ساعة القادمة\n"
                     "نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1 | تصميم: أحمد عمر ظافر"),
                 fontsize=14.5, fontweight='bold', x=cx, ha='center',
                 fontfamily=arabic_fonts, y=0.995)

    fig.text(0.01, 0.004,
             "Scenario product calibrated to experimental GFS v17-HR1 behavior — NOT a dynamical v17 run  |  "
             "Method: Quantile Delta Mapping (Cannon et al. 2015) applied to operational GFS v16.3 cycle "
             f"{ymd}/{cc}Z (APCP cumulative 0–24 h from f024)\n"
             "v17-HR1 climatology scaled to the 24-h window via the reference-sample 24 h/10-day ratio "
             "(gamma shape α unchanged)  |  Reference: the same 24 GFS runs — paired 0–24 h & 0–240 h samples "
             "(23 Aug–12 Sep 2026)\n"
             "v17-HR1 climatology: UFS Replay (NOAA PSL, CC BY 4.0) — 12×10-day Sep–Oct windows 1994–2022  |  "
             "Boundaries: Natural Earth + GADM v4.1  |  Display grid ~3 km (interpolated from 0.25\u00b0 GFS)  |  Experimental guidance only",
             fontsize=6.8, color='#555555', ha='left', va='bottom')

    output = f"hadramout_v17_total_24h_{ymd}_{cc}Z.png"
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    print(f"✅ خريطة الإجمالي 24 ساعة: {output}")

    # ===== تقرير المحطات + CSV =====
    print(f"\n📊 المحطات (دورة {ymd[6:]}/{cc}Z — إجمالي الـ24 ساعة القادمة):")
    print("-" * 58)
    print(f"{'المحطة':<16} {'إجمالي 24س':>10} {'خام 24س':>9}")
    print("-" * 58)
    rows = []
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lats - la).argmin()); j = int(np.abs(lons - lo).argmin())
        vt = float(field[i, j]); vr = float(x0t[i, j])
        rows.append((name, lo, la, vt, vr))
        print(f"  {name:<16} {vt:>9.1f} {vr:>8.1f}")
    print("-" * 58)
    with open(output.replace(".png", "_stations.csv"), "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,scenario_total_24h_mm,raw_total_24h_mm\n")
        for (name, lo, la, vt, vr) in rows:
            f.write(f"{name},{lo:.2f},{la:.2f},{vt:.1f},{vr:.1f}\n")
    print("✅ جدول المحطات حُفظ")


if __name__ == "__main__":
    ymd = sys.argv[1] if len(sys.argv) > 1 else "20260914"
    cc = (sys.argv[2] if len(sys.argv) > 2 else "12").zfill(2)
    main(ymd, cc)
