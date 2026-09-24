# -*- coding: utf-8 -*-
"""
===============================================================================
  خريطة إجمالي أمطار الأيام العشرة القادمة — محافظة حضرموت
  بنمذجة سيناريوهية وفق منهج ودوال النظام التجريبي NOAA GFS v17-HR1
===============================================================================
  المنهجية (Scenario Modeling Chain):
   1) المحرّك الديناميكي : توعُج GFS التشغيلي v16.3 (دورة اليوم 00Z،
                           رسالة APCP التراكمية 0–240 ساعة).
   2) منظومة قيم v17    : المناخ المطرية الموسمية (سبتمبر–أكتوبر) لنظام
                           v17-HR1 من سجل UFS Replay (NOAA PSL) — 12 نافذة
                           عشرية موزعة 1994–2022، كل نافذة = 40 دورة تحليل
                           (GFSFLX.GrbF03، حقل prate) مع عامل اكتمال زمني ×2.
   3) المرجع التشغيلي    : أحدث 24 تشغيلاً لـ v16.3 (23 أغسطس – 12 سبتمبر).
   4) دالة النقل         : Quantile Delta Mapping (Cannon et al. 2015):
                           scen = γ17⁻¹(q) + (x₀ − ref_q)
                           حيث q رتبة توقع اليوم ضمن المرجع، γ17 توزيع گاما
                           لمناخ v17-HR1 (شكل مجمّع + مقياس موضعي).
   ⚠️ هذا سيناريو إحصائي معاير على سلوك v17-HR1 — وليس تشغيلاً ديناميكياً
      للنموذج v17 نفسه (يتطلب حواسيب فائقة وبيانات تهيئة تشغيلية).
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

OUTPUT_MAIN = "hadramout_v17_scenario_20260913_10day.png"
OUTPUT_DIAG = "hadramout_v17_scenario_20260913_diagnostic.png"


def _base_axes(ax, title_en):
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
    ax.set_title(title_en, fontsize=10, fontweight="bold", loc="left")


def _fill(ax, field, lats, lons, levels, colors, cb_label):
    f_fine, la_f, lo_f = _interp_to_fine_grid(field, lats, lons, DISPLAY_RES_DEG)
    f_plot = np.where(f_fine >= RAIN_MIN_MM, f_fine, np.nan)
    cmap = ListedColormap(colors)
    cmap.set_over("#050A3C")
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(levels, cmap.N)
    cf = ax.contourf(lo_f, la_f, f_plot, levels=levels, cmap=cmap, norm=norm,
                     extend="max", alpha=1.0, zorder=2, transform=ccrs.PlateCarree())
    return cf, f_fine


def main():
    d = np.load("gfs_cache/v17replay/scenario_20260913.npz")
    scen, x0, mu17 = d["scen"], d["x0"], d["mu17"]
    q10, q90, lats, lons = d["q10"], d["q90"], d["lat"], d["lon"]

    levels = [0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 140]
    rain_colors = [
        "#7AE1E8", "#0AC5D9", "#28B2DE", "#31A0E2", "#328DE6", "#337AEA",
        "#736FEA", "#9860EB", "#B94AEB", "#CB49D4", "#D64DBB", "#DF51A0",
        "#E65687", "#EB5B6F", "#EF6152", "#F27343", "#F69343", "#F9AF42",
        "#FACA40", "#FAE63C",
    ]

    # ========================== الخريطة الرئيسية ==========================
    arabic_fonts = _setup_arabic_font()
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
    except Exception as e:
        print(f"⚠️ حدود المحافظات: {e}")
    try:
        _add_hadramout_districts(ax)
    except Exception as e:
        print(f"⚠️ حدود المديريات: {e}")

    scen_fine, lats_f, lons_f = _interp_to_fine_grid(scen, lats, lons, DISPLAY_RES_DEG)
    print(f"🗺️ شبكة العرض: {len(lats_f)}×{len(lons_f)} عقدة (≈{DISPLAY_RES_KM:.0f} كم)")
    scen_plot = np.where(scen_fine >= RAIN_MIN_MM, scen_fine, np.nan)

    cmap = ListedColormap(rain_colors)
    cmap.set_over("#050A3C")
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(levels, cmap.N)
    cf = ax.contourf(lons_f, lats_f, scen_plot, levels=levels, cmap=cmap, norm=norm,
                     extend="max", alpha=1.0, zorder=2, transform=ccrs.PlateCarree())
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", shrink=0.80,
                      pad=0.015, aspect=32)
    cb.set_label("Scenario 10-Day Total Precipitation (mm)", fontsize=10)
    cb.set_ticks(levels)
    cb.ax.tick_params(labelsize=7.5)

    # إحصاءات داخل حدود الخريطة
    mlat = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    mlon = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    sub = scen[np.ix_(mlat, mlon)]
    vmax = float(sub.max())
    imax = np.unravel_index(np.argmax(sub), sub.shape)
    vmax_lat = float(lats[mlat][imax[0]])
    vmax_lon = float(lons[mlon][imax[1]])

    # النقاط المرجعية الحيوية
    stations = {
        'Mukalla': (49.12, 14.53), 'Ash Shihr': (49.60, 14.76),
        'Al-Dhabba Port': (49.50, 14.70), 'Shuhayr': (49.42, 14.65),
        'Wadi Huwayrah': (49.35, 14.82), 'Seiyun (Wadi)': (48.78, 15.93),
        'Tarim': (49.00, 16.05), 'Dawan (Wadi)': (48.45, 15.00),
        'Al-Ghaydah': (52.19, 16.21),
    }
    station_rows = []
    for name, (st_lon, st_lat) in stations.items():
        ax.plot(st_lon, st_lat, marker='o', markersize=4.5, color='darkred',
                transform=ccrs.PlateCarree(), zorder=6)
        iy = int(np.abs(lats - st_lat).argmin())
        ix = int(np.abs(lons - st_lon).argmin())
        station_rows.append((name, st_lon, st_lat, float(x0[iy, ix]),
                             float(scen[iy, ix]), float(mu17[iy, ix]),
                             float(q10[iy, ix]), float(q90[iy, ix])))

    ax.text(0.015, 0.985,
            "Forecast window: 13 Sep 2026 00Z → 23 Sep 2026 00Z\n"
            "Driver: GFS v16.3 op (20260913/00Z) | Calibrated to v17-HR1 behavior",
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
            color='#222222', fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.75, edgecolor='#bbbbbb'))
    ax.text(0.985, 0.02,
            f"Scenario max: {vmax:.0f} mm  (near {vmax_lat:.2f}N, {vmax_lon:.2f}E)\n"
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

    plt.title(_ar("محافظة حضرموت — إجمالي الأمطار المتوقع خلال الأيام العشرة القادمة\n"
                  "نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1 (UFS Replay) | دقة العرض ≈ 9 كم"),
              fontsize=12, fontweight='bold', loc='right', pad=12,
              fontfamily=arabic_fonts)

    fig.text(0.09, 0.004,
             "Scenario product calibrated to experimental GFS v17-HR1 behavior — NOT a dynamical v17 run  |  "
             "Method: Quantile Delta Mapping — scen = γ17⁻¹(q) + (x₀ − ref_q)\n"
             "v17-HR1 climatology: UFS Replay (NOAA PSL, CC BY 4.0) via AWS — 12×10-day Sep–Oct windows 1994–2022, "
             "40 analysis cycles each (prate, GFSFLX.GrbF03, ×2 completeness)  |  Reference: latest 24 GFS v16.3 runs (23 Aug–12 Sep 2026)\n"
             "Driver: operational GFS v16.3 20260913/00Z APCP 0–240 h  |  Boundaries: Natural Earth + GADM v4.1  |  Experimental guidance only",
             fontsize=6.8, color='#555555', ha='left', va='bottom')

    plt.savefig(OUTPUT_MAIN, bbox_inches='tight')
    plt.close()
    print(f"✅ الخريطة الرئيسية: {OUTPUT_MAIN}")

    # ========================== اللوحة التشخيصية ==========================
    fig2 = plt.figure(figsize=(16.5, 6.2), dpi=130)
    for k, (field, ttl) in enumerate([
            (x0, "1) Driver — operational GFS v16.3 (raw 10-day total)"),
            (scen, "2) Scenario — v17-HR1 behavior (Quantile Delta Mapping)"),
            (q90, "3) v17-HR1 Sep–Oct climatological q90 envelope")]):
        axk = plt.subplot(1, 3, k + 1, projection=ccrs.PlateCarree())
        axk.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        _base_axes(axk, ttl)
        cfk, _ = _fill(axk, field, lats, lons, levels, rain_colors, "")
        cbk = fig2.colorbar(cfk, ax=axk, orientation="horizontal", shrink=0.82,
                            pad=0.06, aspect=26)
        cbk.set_ticks(levels)
        cbk.ax.tick_params(labelsize=6.5)
        cbk.set_label("mm", fontsize=8)
    fig2.suptitle("Hadramout — next 10 days (13→23 Sep 2026): v17-HR1 scenario modeling diagnostics",
                  fontsize=12, fontweight="bold")
    plt.savefig(OUTPUT_DIAG, bbox_inches='tight')
    plt.close()
    print(f"✅ اللوحة التشخيصية: {OUTPUT_DIAG}")

    # ========================== تقرير المحطات ==========================
    print(f"\n📊 النقاط المرجعية — خام v16.3 مقابل سيناريو v17-HR1:")
    print("-" * 78)
    print(f"{'المحطة':<16} {'خام v16.3':>10} {'سيناريو v17':>12} {'مناخ v17':>9} {'نطاق 10-90%':>12}")
    print("-" * 78)
    for name, lon, lat, v_raw, v_scen, v_mu, v_q10, v_q90 in station_rows:
        bar = "█" * int(round(min(v_scen, 60) / 4))
        print(f"  {name:<16} {v_raw:>8.1f} {v_scen:>10.1f} {v_mu:>8.1f} [{v_q10:>4.0f}-{v_q90:>4.0f}]  {bar}")
    print("-" * 78)

    with open(OUTPUT_MAIN.replace(".png", "_stations.csv"), "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,raw_v16_3_mm,scenario_v17_qdm_mm,"
                "v17_clim_mean_mm,v17_clim_q10_mm,v17_clim_q90_mm\n")
        for name, lon, lat, v_raw, v_scen, v_mu, v_q10, v_q90 in station_rows:
            f.write(f"{name},{lon:.2f},{lat:.2f},{v_raw:.1f},{v_scen:.1f},"
                    f"{v_mu:.1f},{v_q10:.1f},{v_q90:.1f}\n")
    print("✅ جدول المحطات حُفظ")


if __name__ == "__main__":
    main()
