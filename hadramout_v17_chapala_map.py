# -*- coding: utf-8 -*-
"""
===============================================================================
  خريطة تراكم أمطار إعصار تشابالا فوق محافظة حضرموت
  من بيانات نظام GFS v17-HR1 التجريبي (UFS Replay — NOAA PSL)
===============================================================================
  المصدر       : GEFSv13/GFSv17-HR1 UFS Replay — إعادة تشغيل النظام المزدوج
                 بفيزياء v17-HR1 (Thompson-Eidhammer، Noah-MP) وديناميكا
                 مقيدة بإعادة تحليل ERA5 (عبر IAU).
                 NOAA Physical Sciences Laboratory + NOAA Open Data Dissemination
                 (ترخيص CC BY 4.0 — https://psl.noaa.gov/data/ufs_replay/)
  المتغير      : prate — معدل التساقط (متوسط نافذة 3 ساعات، kg m⁻² s⁻¹)
                 من ملفات GFSFLX.GrbF03 لدورات التحليل السداسية
                 (AWS: noaa-ufs-gefsv13replay-pds)
  الحدث        : إعصار تشابالا — اليابسة قرب المكلا في 3 نوفمبر 2015
  النافذة     : 2015-10-29 → 2015-11-08 (10 أيام؛ مجموع 40 نافذة قياس × 3 ساعات
                 = 120 ساعة من أصل 240، لأن نوافذ النصف الأول من كل دورة تحليل
                 لا تنشر حقل prate في ملفات GRIB)
  ملاحظة      : هذه بيانات تجريبية لنظام ما قبل GFSv17 التشغيلي — وليست
                 النسخة المتوازية الحية (غير منشورة علناً حتى تنفيذ v17).
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

# إعادة استخدام دوال الخريطة المعتمدة من السكربت الرئيسي
from hadramout_gfs_10day_rain import (
    _ar, _setup_arabic_font, _add_yemen_admin1, _add_hadramout_districts,
    _interp_to_fine_grid, MAP_EXTENT, RAIN_MIN_MM, DISPLAY_RES_DEG, DISPLAY_RES_KM,
)

OUTPUT_IMG = "hadramout_v17_chapala_10day.png"


def main():
    d = np.load("gfs_cache/v17replay/chapala_10day.npz")
    precip, lats, lons = d["acc"], d["lat"], d["lon"]
    print(f"البيانات: {precip.shape} | الأعلى: {precip.max():.1f} ملم")

    arabic_fonts = _setup_arabic_font()
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # ---------------------- خلفية جغرافية ----------------------
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

    # ---------------------- طبقة المطر ----------------------
    precip_fine, lats_f, lons_f = _interp_to_fine_grid(
        precip, lats, lons, DISPLAY_RES_DEG)
    print(f"🗺️ شبكة العرض: {len(lats_f)}×{len(lons_f)} عقدة (≈{DISPLAY_RES_KM:.0f} كم)")
    precip_plot = np.where(precip_fine >= RAIN_MIN_MM, precip_fine, np.nan)

    levels = [0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 140]
    rain_colors = [
        "#7AE1E8", "#0AC5D9", "#28B2DE", "#31A0E2", "#328DE6", "#337AEA",
        "#736FEA", "#9860EB", "#B94AEB", "#CB49D4", "#D64DBB", "#DF51A0",
        "#E65687", "#EB5B6F", "#EF6152", "#F27343", "#F69343", "#F9AF42",
        "#FACA40", "#FAE63C",
    ]
    cmap = ListedColormap(rain_colors)
    cmap.set_over("#050A3C")
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(levels, cmap.N)

    cf = ax.contourf(lons_f, lats_f, precip_plot, levels=levels, cmap=cmap,
                     norm=norm, extend="max", alpha=1.0, zorder=2,
                     transform=ccrs.PlateCarree())
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", shrink=0.80,
                      pad=0.015, aspect=32)
    cb.set_label("Total 10-Day Precipitation (mm)", fontsize=10)
    cb.set_ticks(levels)
    cb.ax.tick_params(labelsize=7.5)

    # ---------------------- إحصاءات ----------------------
    mlat = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    mlon = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    sub = precip[np.ix_(mlat, mlon)]
    vmax = float(sub.max())
    vmean = float(sub.mean())
    imax = np.unravel_index(np.argmax(sub), sub.shape)
    vmax_lat = float(lats[mlat][imax[0]])
    vmax_lon = float(lons[mlon][imax[1]])

    # ---------------------- النقاط المرجعية الحيوية ----------------------
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
        station_rows.append((name, st_lon, st_lat, float(precip[iy, ix])))

    # ---------------------- صندوقا المعلومات ----------------------
    ax.text(0.015, 0.985,
            "Event: Cyclone Chapala (landfall near Mukalla, 3 Nov 2015)\n"
            "Window: 29 Oct 2015 00Z → 8 Nov 2015 00Z",
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
            color='#222222', fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.75, edgecolor='#bbbbbb'))
    ax.text(0.985, 0.02,
            f"Domain max: {vmax:.0f} mm  (near {vmax_lat:.2f}N, {vmax_lon:.2f}E)\n"
            f"Domain mean: {vmean:.1f} mm  |  Display grid: ~9 km (from 0.25°)",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            color='#222222', zorder=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75,
                      edgecolor='#bbbbbb'))

    # ---------------------- شبكة الإحداثيات ----------------------
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': '#333333'}
    gl.ylabel_style = {'size': 8, 'color': '#333333'}

    # ---------------------- العنوان ----------------------
    plt.title(_ar("محافظة حضرموت — إجمالي تراكم الأمطار خلال 10 أيام\n"
                  "إعصار تشابالا | النظام التجريبي NOAA GFS v17-HR1 (UFS Replay) | دقة العرض ≈ 9 كم"),
              fontsize=12.5, fontweight='bold', loc='right', pad=12,
              fontfamily=arabic_fonts)

    fig.text(0.09, 0.004,
             "Data: NOAA PSL UFS Replay (GEFSv13/GFSv17-HR1, experimental — CC BY 4.0, psl.noaa.gov/data/ufs_replay) via AWS noaa-ufs-gefsv13replay-pds  |  "
             "Variable: prate, 0–3 h mean per 6-hourly cycle (GFSFLX.GrbF03); 10-day total = 40 windows (120 h measured)\n"
             "Replay: v17-HR1 physics (FV3 coupled, Thompson-Eidhammer, Noah-MP) with dynamics constrained to ERA5 (IAU)  |  "
             "Admin boundaries: Natural Earth + GADM v4.1  |  Station values = nearest 0.25° grid point",
             fontsize=7, color='#555555', ha='left', va='bottom')

    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    plt.close()
    print(f"✅ تم تصدير الخريطة: {OUTPUT_IMG}")

    # ---------------------- تقرير المحطات ----------------------
    print("\n📊 التراكم عند النقاط المرجعية (إعصار تشابالا — نظام v17-HR1):")
    print("-" * 64)
    for name, lon, lat, v in station_rows:
        bar = "█" * int(round(min(v, 160) / 8))
        print(f"  {name:<16} ({lat:5.2f}N, {lon:5.2f}E) : {v:7.1f} mm  {bar}")
    print("-" * 64)
    with open(OUTPUT_IMG.replace(".png", "_stations.csv"), "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,precip_mm_chapala_10day\n")
        for name, lon, lat, v in station_rows:
            f.write(f"{name},{lon:.2f},{lat:.2f},{v:.1f}\n")
    print("✅ جدول المحطات حُفظ")


if __name__ == "__main__":
    main()
