# -*- coding: utf-8 -*-
"""
===============================================================================
  خريطة إجمالي أمطار السحب الرعدية (الحملية) الممطرة — الأيام العشرة القادمة
  محافظة حضرموت — بنمذجة سيناريوهية وفق منهج ودوال NOAA GFS v17-HR1
===============================================================================
  المتغير      : التساقط الحملي (Convective Precipitation — ACPCP/CPRAT)
  المحرك       : GFS التشغيلي — دورة 2026-09-13 06Z (آخر تحديث، نافذة
                 13 سبتمبر 06Z → 23 سبتمبر 06Z، رسالة ACPCP التراكمية 0–240س)
  المنهجية     : نفس سلسلة QDM (Cannon et al. 2015):
                 scen_conv = γ17c⁻¹(q_conv) + (x₀c − ref_q_conv)
                 حيث مناخ v17-HR1 الحملي = مناخ v17-HR1 الكلي (من سجل UFS
                 Replay عبر GRIB) × النسبة الحملية المرصودة من عينة الـ24
                 تشغيلاً المرجعية (82.6% إقليمياً).
  ملاحظة تقنية : حقول المطر الحملي في مخرجات UFS Replay (cprat_ave وغيرها)
                 معطوبة في كل الصيغ (تم التحقق) — لذا نُقل النسبة الحملية من
                 العينة المرجعية التشغيلية، مع بقاء منظومة القيم الكلية
                 والشكل الإحصائي (گاما) من نظام v17-HR1 الفعلي.
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

OUTPUT_OP = "hadramout_conv_20260913_06Z.png"
OUTPUT_SC = "hadramout_v17_conv_20260913_06Z_10day.png"
OUTPUT_DIAG = "hadramout_v17_conv_20260913_06Z_diagnostic.png"

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


def _draw_map(field, lats, lons, arabic_fonts, title_ar, box_texts, footer,
              out_path, cb_label):
    """رسم خريطة واحدة بالتصميم المعتمد"""
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d4e4f0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f2e8", zorder=0)
    ax.coastlines("50m", linewidth=0.8, color="#515151", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.0,
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
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", shrink=0.80,
                      pad=0.015, aspect=32)
    cb.set_label(cb_label, fontsize=10)
    cb.set_ticks(LEVELS)
    cb.ax.tick_params(labelsize=7.5)

    # إحصاءات النطاق
    mlat = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    mlon = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    sub = field[np.ix_(mlat, mlon)]
    vmax = float(sub.max())
    imax = np.unravel_index(np.argmax(sub), sub.shape)
    vlat = float(lats[mlat][imax[0]]); vlon = float(lons[mlon][imax[1]])

    for name, (st_lon, st_lat) in STATIONS.items():
        ax.plot(st_lon, st_lat, marker='o', markersize=4.5, color='darkred',
                transform=ccrs.PlateCarree(), zorder=6)

    ax.text(0.015, 0.985, box_texts[0], transform=ax.transAxes, ha='left',
            va='top', fontsize=8.5, color='#222222', fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.75, edgecolor='#bbbbbb'))
    ax.text(0.985, 0.02,
            f"Domain max: {vmax:.0f} mm  (near {vlat:.2f}N, {vlon:.2f}E)\n"
            f"Domain mean: {sub.mean():.1f} mm  |  Display grid: ~9 km (from 0.25°)",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            color='#222222', zorder=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75,
                      edgecolor='#bbbbbb'))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': '#333333'}
    gl.ylabel_style = {'size': 8, 'color': '#333333'}

    plt.title(_ar(title_ar), fontsize=12, fontweight='bold', loc='right', pad=12,
              fontfamily=arabic_fonts)
    fig.text(0.09, 0.004, footer, fontsize=6.8, color='#555555',
             ha='left', va='bottom')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"✅ {out_path}")
    return vmax


def main():
    d = np.load("gfs_cache/v17replay/scenario_conv_2026091306.npz")
    scen_conv, x0c = d["scen_conv"], d["x0c"]
    scen_tot, x0t = d["scen_tot"], d["x0t"]
    q10, q90, lats, lons = d["q10"], d["q90"], d["lat"], d["lon"]

    # سقف الاتساق الفيزيائي: الحملي ≤ الكلي في السيناريو
    scen_conv = np.minimum(scen_conv, scen_tot)

    arabic_fonts = _setup_arabic_font()

    # ===== 1) الخريطة التشغيلية الخام (ACPCP 06Z) =====
    _draw_map(
        x0c, lats, lons, arabic_fonts,
        "محافظة حضرموت — إجمالي أمطار السحب الرعدية (الحملية) خلال الأيام العشرة القادمة\n"
        "المخرج الخام لنموذج NOAA GFS v16.3 التشغيلي — دورة 13 سبتمبر 06Z | دقة العرض ≈ 9 كم",
        ("Variable: ACPCP (convective) — 0–240 h cumulative\n"
         "Cycle: 13 Sep 2026 06Z → 23 Sep 2026 06Z (latest update)"),
        "Data: NOAA/NCEP operational GFS v16.3, cycle 20260913/06Z — ACPCP surface (convective precip, "
        "cumulative 0–240 h) via NOMADS/AWS  |  Display grid ~9 km (bilinear of 0.25°)  |  "
        "Boundaries: Natural Earth + GADM v4.1",
        OUTPUT_OP, "10-Day Convective Precipitation (mm)")

    # ===== 2) خريطة سيناريو v17-HR1 الحملي (المنتج الرئيسي) =====
    _draw_map(
        scen_conv, lats, lons, arabic_fonts,
        "محافظة حضرموت — إجمالي أمطار السحب الرعدية (الحملية) الممطرة خلال الأيام العشرة القادمة\n"
        "نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1 (UFS Replay) | دورة 13 سبتمبر 06Z | دقة العرض ≈ 9 كم",
        ("Forecast window: 13 Sep 2026 06Z → 23 Sep 2026 06Z\n"
         "Driver: GFS v16.3 op ACPCP | Calibrated to v17-HR1 behavior (QDM)"),
        "Scenario product calibrated to experimental GFS v17-HR1 behavior — NOT a dynamical v17 run  |  "
        "Method: QDM on ACPCP — scen = γ17c⁻¹(q) + (x₀c − ref_q); v17-HR1 conv climatology = replay total "
        "climatology × reference convective fraction (82.6% regional; replay convective fields corrupt — verified)\n"
        "v17-HR1 total climatology: UFS Replay (NOAA PSL, CC BY 4.0) — 12×10-day Sep–Oct windows 1994–2022  |  "
        "Reference: latest 24 GFS runs (23 Aug–12 Sep 2026)  |  Boundaries: Natural Earth + GADM v4.1  |  "
        "Experimental guidance only",
        OUTPUT_SC, "Scenario 10-Day Convective Precipitation (mm)")

    # ===== 3) اللوحة التشخيصية =====
    fig2 = plt.figure(figsize=(16.5, 6.2), dpi=130)
    panels = [
        (x0c, "1) Driver raw — operational ACPCP v16.3 (13/06Z)"),
        (scen_conv, "2) Scenario — v17-HR1 convective (QDM)"),
        (scen_tot, "3) Scenario — v17-HR1 total precip (for reference)"),
    ]
    for k, (field, ttl) in enumerate(panels):
        axk = plt.subplot(1, 3, k + 1, projection=ccrs.PlateCarree())
        axk.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        axk.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d4e4f0", zorder=0)
        axk.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f2e8", zorder=0)
        axk.coastlines("50m", linewidth=0.7, color="#515151", zorder=3)
        try:
            _add_hadramout_districts(axk)
        except Exception:
            pass
        axk.set_title(ttl, fontsize=9.5, fontweight="bold", loc="left")
        f_fine, la_f, lo_f = _interp_to_fine_grid(field, lats, lons, DISPLAY_RES_DEG)
        f_plot = np.where(f_fine >= RAIN_MIN_MM, f_fine, np.nan)
        cmap = ListedColormap(RAIN_COLORS); cmap.set_over("#050A3C"); cmap.set_bad(alpha=0.0)
        cfk = axk.contourf(lo_f, la_f, f_plot, levels=LEVELS, cmap=cmap,
                           norm=BoundaryNorm(LEVELS, cmap.N), extend="max",
                           alpha=1.0, zorder=2, transform=ccrs.PlateCarree())
        cbk = fig2.colorbar(cfk, ax=axk, orientation="horizontal", shrink=0.82,
                            pad=0.06, aspect=26)
        cbk.set_ticks(LEVELS); cbk.ax.tick_params(labelsize=6.5); cbk.set_label("mm", fontsize=8)
    fig2.suptitle("Hadramout — next 10 days (13→23 Sep 2026, 06Z): convective (thunderstorm) rain "
                  "— v17-HR1 scenario diagnostics", fontsize=11.5, fontweight="bold")
    plt.savefig(OUTPUT_DIAG, bbox_inches='tight')
    plt.close()
    print(f"✅ {OUTPUT_DIAG}")

    # ===== 4) تقرير المحطات =====
    print(f"\n📊 النقاط المرجعية — الحملي (السحب الرعدية) خام v16.3 مقابل سيناريو v17-HR1:")
    print("-" * 74)
    print(f"{'المحطة':<16} {'حملي خام':>9} {'سيناريو حملي':>13} {'سيناريو كلي':>12} {'حصة حملية':>9}")
    print("-" * 74)
    rows = []
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lats - la).argmin()); j = int(np.abs(lons - lo).argmin())
        v_raw, v_sc, v_tot = float(x0c[i, j]), float(scen_conv[i, j]), float(scen_tot[i, j])
        share = v_sc / v_tot if v_tot > 0.5 else float("nan")
        rows.append((name, lo, la, v_raw, v_sc, v_tot, share))
        bar = "█" * int(round(min(v_sc, 40) / 2.5))
        print(f"  {name:<16} {v_raw:>7.1f} {v_sc:>11.1f} {v_tot:>10.1f} {share:>8.0%}  {bar}")
    print("-" * 74)

    with open(OUTPUT_SC.replace(".png", "_stations.csv"), "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,raw_conv_v16_3_mm,scenario_v17_conv_mm,"
                "scenario_v17_total_mm,conv_share\n")
        for name, lo, la, v_raw, v_sc, v_tot, share in rows:
            f.write(f"{name},{lo:.2f},{la:.2f},{v_raw:.1f},{v_sc:.1f},{v_tot:.1f},{share:.2f}\n")
    print("✅ جدول المحطات حُفظ")


if __name__ == "__main__":
    main()
