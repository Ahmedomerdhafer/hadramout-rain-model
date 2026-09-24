# -*- coding: utf-8 -*-
"""
===============================================================================
  لوحة مزدوجة: إجمالي الأمطار + الأمطار الرعدية (الحملية) — الأيام العشرة القادمة
  محافظة حضرموت — نمذجة سيناريوهية وفق دوال NOAA GFS v17-HR1 التجريبي
===============================================================================
  الدورة        : 2026-09-13 06Z (أحدث دورة مكتملة عند الإنتاج)
                  النافذة: 13 سبتمبر 06Z → 23 سبتمبر 06Z
  اللوحة اليسرى : إجمالي الأمطار (كلي — APCP → QDM على منظومة v17-HR1)
  اللوحة اليمنى : الأمطار الرعدية/الحملية (ACPCP → QDM بمناخ حملي مقيّد
                  بالنسبة الحملية المرصودة 82.6% من العينة المرجعية)
  المنهجية      : نفس سلسلة النمذجة المعتمدة (Cannon et al. 2015)
===============================================================================
"""

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
    _interp_to_fine_grid, MAP_EXTENT, RAIN_MIN_MM, DISPLAY_RES_DEG, DISPLAY_RES_KM,
)

OUTPUT = "hadramout_v17_total_conv_20260913_06Z_panel.png"

LEVELS = [0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 140]
RAIN_COLORS = [
    "#7AE1E8", "#0AC5D9", "#28B2DE", "#31A0E2", "#328DE6", "#337AEA",
    "#736FEA", "#9860EB", "#B94AEB", "#CB49D4", "#D64DBB", "#DF51A0",
    "#E65687", "#EB5B6F", "#EF6152", "#F27343", "#F69343", "#F9AF42",
    "#FACA40", "#FAE63C",
]

STATIONS = {
    'Mukalla': (49.12, 14.53), 'Ash Shihr': (49.60, 14.76),
    'Al-Dhabba Port': (49.50, 14.70), 'Shuhayr': (49.42, 14.65),
    'Wadi Huwayrah': (49.35, 14.82), 'Seiyun (Wadi)': (48.78, 15.93),
    'Tarim': (49.00, 16.05), 'Dawan (Wadi)': (48.45, 15.00),
    'Al-Ghaydah': (52.19, 16.21),
}


def _panel(ax, field, lats, lons, arabic_fonts, title_ar, title_en):
    """رسم لوحة واحدة ضمن اللوحة المزدوجة"""
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
    f_plot = np.where(f_fine >= RAIN_MIN_MM, f_fine, np.nan)
    cmap = ListedColormap(RAIN_COLORS)
    cmap.set_over("#050A3C")
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(LEVELS, cmap.N)
    cf = ax.contourf(lons_f, lats_f, f_plot, levels=LEVELS, cmap=cmap, norm=norm,
                     extend="max", alpha=1.0, zorder=2, transform=ccrs.PlateCarree())

    # إحصاءات النطاق
    mlat = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    mlon = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    sub = field[np.ix_(mlat, mlon)]
    vmax = float(sub.max())
    imax = np.unravel_index(np.argmax(sub), sub.shape)
    vlat = float(lats[mlat][imax[0]]); vlon = float(lons[mlon][imax[1]])

    for st_lon, st_lat in STATIONS.values():
        ax.plot(st_lon, st_lat, marker='o', markersize=4, color='darkred',
                transform=ccrs.PlateCarree(), zorder=6)

    ax.text(0.985, 0.02,
            f"max: {vmax:.0f} mm ({vlat:.2f}N, {vlon:.2f}E)\nmean: {sub.mean():.1f} mm",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
            color='#222222', zorder=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75,
                      edgecolor='#bbbbbb'))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.45, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7, 'color': '#333333'}
    gl.ylabel_style = {'size': 7, 'color': '#333333'}

    ax.set_title(f"{title_en}\n{_ar(title_ar)}", fontsize=11, fontweight='bold',
                 loc='center', pad=8, fontfamily=arabic_fonts)
    return cf


def main():
    d = np.load("gfs_cache/v17replay/scenario_conv_2026091306.npz")
    lats, lons = d["lat"], d["lon"]
    scen_tot = d["scen_tot"]
    scen_conv = np.minimum(d["scen_conv"], scen_tot)   # الاتساق الفيزيائي
    x0t, x0c = d["x0t"], d["x0c"]

    arabic_fonts = _setup_arabic_font()
    fig = plt.figure(figsize=(18, 8.8), dpi=150)

    # ===== اللوحة اليسرى: إجمالي الأمطار =====
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    cf1 = _panel(ax1, scen_tot, lats, lons, arabic_fonts,
                 "إجمالي الأمطار (كلي)",
                 "Total Precipitation — v17-HR1 scenario")

    # ===== اللوحة اليمنى: الأمطار الرعدية الحملية =====
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    cf2 = _panel(ax2, scen_conv, lats, lons, arabic_fonts,
                 "الأمطار الرعدية (حملي)",
                 "Convective (Thunderstorm) Rain — v17-HR1 scenario")

    # ===== شريط ألوان مشترك =====
    cb = fig.colorbar(cf2, ax=[ax1, ax2], orientation="vertical",
                      shrink=0.82, pad=0.015, aspect=34)
    cb.set_label("10-Day Total Precipitation (mm)", fontsize=10)
    cb.set_ticks(LEVELS)
    cb.ax.tick_params(labelsize=7.5)

    # صندوق النافذة الزمنية (فوق اللوحة اليسرى)
    ax1.text(0.015, 0.985,
             "Window: 13 Sep 2026 06Z → 23 Sep 2026 06Z (10 days)\n"
             "Cycle: 20260913/06Z — latest complete run",
             transform=ax1.transAxes, ha='left', va='top', fontsize=8.5,
             color='#222222', fontweight='bold', zorder=8,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.75, edgecolor='#bbbbbb'))

    # ===== العنوان الرئيسي (موسّط فوق الخريطتين) + اسم المصمم =====
    # التوسيط على مركز منطقة اللوحتين الفعلية (لا مركز الشكل كاملاً)،
    # لأن شريط الألوان الجانبي يزاحة الخرائط يساراً فيبدو العنوان منحرفاً
    p1, p2 = ax1.get_position(), ax2.get_position()
    cx = (p1.x0 + p2.x1) / 2.0
    fig.suptitle(_ar("محافظة حضرموت — إجمالي الأمطار والأمطار الرعدية الممطرة خلال الأيام العشرة القادمة\n"
                     "نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1 (UFS Replay) | دقة العرض ≈ 9 كم"),
                 fontsize=13.5, fontweight='bold', x=cx, ha='center',
                 fontfamily=arabic_fonts, y=0.995)
    fig.text(cx, 0.938, _ar("تصميم: أحمد عمر ظافر"),
             fontsize=11.5, fontweight='bold', ha='center', va='top',
             fontfamily=arabic_fonts, color='#4a3b28')

    # ===== الحاشية =====
    fig.text(0.01, 0.003,
             "Scenario products calibrated to experimental GFS v17-HR1 behavior — NOT dynamical v17 runs  |  "
             "Method: Quantile Delta Mapping (Cannon et al. 2015) applied to operational GFS v16.3 cycle 20260913/06Z "
             "(APCP total & ACPCP convective, cumulative 0–240 h)\n"
             "v17-HR1 climatology: UFS Replay (NOAA PSL, CC BY 4.0) — 12×10-day Sep–Oct windows 1994–2022  |  "
             "Convective climatology = replay total × reference convective fraction (82.6% regional; replay convective fields corrupt — verified)  |  "
             "Reference: latest 24 GFS runs (23 Aug–12 Sep 2026)\n"
             "Boundaries: Natural Earth (governorates) + GADM v4.1 (districts)  |  "
             "Display grid ~9 km (bilinear of 0.25°)  |  Experimental guidance only",
             fontsize=6.6, color='#555555', ha='left', va='bottom')

    plt.savefig(OUTPUT, bbox_inches='tight')
    plt.close()
    print(f"✅ اللوحة المزدوجة: {OUTPUT}")

    # ===== تقرير المحطات =====
    print(f"\n📊 النقاط المرجعية (دورة 13/06Z — سيناريو v17-HR1):")
    print("-" * 76)
    print(f"{'المحطة':<16} {'إجمالي (كلي)':>12} {'رعدي (حملي)':>12} {'حصة حملية':>9} {'كلي خام':>9} {'حملي خام':>9}")
    print("-" * 76)
    rows = []
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lats - la).argmin()); j = int(np.abs(lons - lo).argmin())
        vt, vc = float(scen_tot[i, j]), float(scen_conv[i, j])
        share = vc / vt if vt > 0.5 else float("nan")
        rows.append((name, lo, la, vt, vc, share,
                     float(x0t[i, j]), float(x0c[i, j])))
        print(f"  {name:<16} {vt:>10.1f} {vc:>10.1f} {share:>8.0%} {x0t[i,j]:>9.1f} {x0c[i,j]:>9.1f}")
    print("-" * 76)

    with open(OUTPUT.replace(".png", "_stations.csv"), "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,scenario_v17_total_mm,scenario_v17_conv_mm,"
                "conv_share,raw_v16_3_total_mm,raw_v16_3_conv_mm\n")
        for name, lo, la, vt, vc, share, rt, rc in rows:
            f.write(f"{name},{lo:.2f},{la:.2f},{vt:.1f},{vc:.1f},{share:.2f},{rt:.1f},{rc:.1f}\n")
    print("✅ جدول المحطات حُفظ")


if __name__ == "__main__":
    main()
