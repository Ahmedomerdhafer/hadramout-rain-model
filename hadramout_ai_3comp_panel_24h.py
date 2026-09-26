# -*- coding: utf-8 -*-
"""
===============================================================================
  لوحة ثلاثية: إجمالي الأمطار + الأمطار الرعدية (الحملية) + أمطار السحب
  المنخفضة (الطبقية غير الحملية) — الـ24 ساعة القادمة — محافظة حضرموت
  نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1
===============================================================================
  الدورة         : 2026-09-15 06Z (أحدث دورة عند الإنتاج — f024 نهائي)
                   النافذة: الـ24 ساعة القادمة (15 سبتمبر 06Z → 16 سبتمبر 06Z)
  اللوحة 1 (يسار) : إجمالي الأمطار (APCP → QDM على منظومة v17-HR1)
  اللوحة 2 (وسط)  : الأمطار الرعدية الحملية (ACPCP → QDM بمناخ حملي مقيّد
                    بالنسبة الحملية المرصودة 82.6%)
  اللوحة 3 (يمين) : أمطار السحب المنخفضة الطبقية = المكوّن غير الحملي
                    (APCP − ACPCP) — في هذه المنطقة هو مطر السحب المنخفضة
                    الساحلية (الركامي الطبقي والطبقيات والرذاذ)
  ملاحظة علمية   : GFS لا يُخرج متغيراً منفصلاً لمطر السحب المنخفضة؛
                    التفكيك المعياري (كلي = حملي + طبقي) هو الأساس الفيزيائي
                    للوحة الثالثة، والاتساق مضمون بالبناء (طبقي = كلي − حملي).
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
from district_labels import add_district_labels

# دقة عرض أدق بطلب المستخدم: ~3 كم (كانت ~9 كم) — استيفاء من بيانات GFS 0.25°
DISPLAY_RES_DEG = 3.0 / 111.32

OUTPUT = "hadramout_ai_3comp_24h_20260924_18Z_panel.png"

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
# يبقى التحقق الدائم أمام رصد مطار الريان (OYRN) عبر verify_mukalla.py.
CONSERVATIVE_FACTOR = 1.0


def _panel(ax, field, lats, lons, arabic_fonts, title_ar, title_en):
    """رسم لوحة واحدة ضمن اللوحة الثلاثية"""
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

    ax.set_title(f"{title_en}\n{_ar(title_ar)}", fontsize=10.5, fontweight='bold',
                 loc='center', pad=8, fontfamily=arabic_fonts)
    # أسماء المديريات — في نهاية اللوحة (تحتاج transData نهائياً)
    add_district_labels(ax, _ar, arabic_fonts, fontsize=6.5, extent=MAP_EXTENT)
    return cf


def main():
    d = np.load("gfs_cache/v17replay/scenario_3comp_24h_ai_2026092418.npz")
    lats, lons = d["lat"], d["lon"]
    scen_tot = d["scen_tot"].astype(np.float64) * CONSERVATIVE_FACTOR
    scen_conv = d["scen_conv"].astype(np.float64) * CONSERVATIVE_FACTOR
    scen_strat = d["scen_strat"].astype(np.float64) * CONSERVATIVE_FACTOR
    x0t, x0c, x0s = d["x0t"], d["x0c"], d["x0s"]

    arabic_fonts = _setup_arabic_font()
    fig = plt.figure(figsize=(24, 11.0), dpi=150)   # نسبة تسمح بامتلاء الخرائط دون تقليل aspect

    panels = [
        (scen_tot, "إجمالي الأمطار (كلي)",
         "Total Precipitation — v17-HR1 scenario"),
        (scen_conv, "الأمطار الرعدية (حملي)",
         "Convective (Thunderstorm) Rain — v17-HR1 scenario"),
        (scen_strat, "أمطار السحب المنخفضة (طبقي)",
         "Low-Cloud (Stratiform) Rain — v17-HR1 scenario"),
    ]
    axes = []
    cf_last = None
    for k, (field, tar, ten) in enumerate(panels):
        axk = plt.subplot(1, 3, k + 1, projection=ccrs.PlateCarree())
        axes.append(axk)
        cf_last = _panel(axk, field, lats, lons, arabic_fonts, tar, ten)

    # مفتاح أفقي مشترك بطراز WeatherBELL (مثل الصورة المرفقة)
    fig.subplots_adjust(left=0.022, right=0.99, top=0.90, bottom=0.225, wspace=0.04)
    p1, p3 = axes[0].get_position(), axes[2].get_position()
    draw_wxbell_legend(fig, x0=p1.x0, x1=p3.x1, y_bar=0.150, bar_h=0.022,
                       num_fs=9.0, unit_fs=10.0, arabic_font=arabic_fonts, ar=_ar)

    # صندوق النافذة الزمنية (فوق اللوحة الأولى)
    axes[0].text(0.015, 0.985,
                 "Window: next 24 h — 24 Sep 2026 18Z → 25 Sep 2026 18Z\n"
                 "Cycle: 20260924/18Z — AI-GFS (GraphCast), f024 final",
                 transform=axes[0].transAxes, ha='left', va='top', fontsize=8.5,
                 color='#222222', fontweight='bold', zorder=8,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                           alpha=0.75, edgecolor='#bbbbbb'))

    # ===== العنوان الرئيسي (موسّط فوق الخرائط الثلاث) + اسم المصمم =====
    p1, p3 = axes[0].get_position(), axes[2].get_position()
    cx = (p1.x0 + p3.x1) / 2.0
    fig.suptitle(_ar("محافظة حضرموت — إجمالي الأمطار والأمطار الرعدية وأمطار السحب المنخفضة خلال الـ24 ساعة القادمة\n"
                     "النموذج الذكي NOAA AI-GFS (GraphCast) مُعالَج بمنهج GFS v17-HR1 | تصميم: أحمد عمر ظافر"),
                 fontsize=14, fontweight='bold', x=cx, ha='center',
                 fontfamily=arabic_fonts, y=0.995)

    # ===== الحاشية =====
    fig.text(0.01, 0.003,
             "Scenario products calibrated to experimental GFS v17-HR1 behavior — NOT dynamical v17 runs  |  "
             "Method: Quantile Delta Mapping (Cannon et al. 2015) applied to NOAA AIGFS v1.0 (GraphCast) cycle 20260924/18Z "
             "(APCP total & ACPCP convective, cumulative 0–24 h)\n"
             "v17-HR1 climatology scaled to the 24-h window via the reference-sample 24 h/10-day ratio (shape α unchanged)\n"
             "Low-cloud (stratiform) rain = non-convective component (APCP − ACPCP) — GFS has no separate low-cloud-rain variable; "
             "in this region it represents coastal stratocumulus/stratus drizzle & nimbostratus  |  "
             "Convective climatology = replay total × reference convective fraction (82.6% regional)\n"
             "v17-HR1 climatology: UFS Replay (NOAA PSL, CC BY 4.0) — 12×10-day Sep–Oct windows 1994–2022  |  "
             "Reference: the same 24 GFS runs — paired 0–24 h & 0–240 h samples (23 Aug–12 Sep 2026)  |  Boundaries: Natural Earth + GADM v4.1  |  "
             "Display grid ~3 km (interpolated from 0.25\u00b0 GFS)  |  Experimental guidance only",
             fontsize=6.6, color='#555555', ha='left', va='bottom')

    plt.savefig(OUTPUT, bbox_inches='tight')
    plt.close()
    print(f"✅ اللوحة الثلاثية: {OUTPUT}")

    # ===== تقرير المحطات =====
    print(f"\n📊 النقاط المرجعية (دورة 24/00Z — AI-GFS — نافذة 24 ساعة):")
    print("-" * 88)
    print(f"{'المحطة':<16} {'إجمالي':>7} {'رعدي':>7} {'منخفض':>7} {'حصة حملية':>9}   {'خام (كلي/حملي/طبقي)':>22}")
    print("-" * 88)
    rows = []
    for name, (lo, la) in STATIONS.items():
        i = int(np.abs(lats - la).argmin()); j = int(np.abs(lons - lo).argmin())
        vt, vc, vs = float(scen_tot[i, j]), float(scen_conv[i, j]), float(scen_strat[i, j])
        share = vc / vt if vt > 0.5 else float("nan")
        rows.append((name, lo, la, vt, vc, vs, share))
        print(f"  {name:<16} {vt:>5.1f} {vc:>5.1f} {vs:>5.1f} {share:>8.0%}   ({x0t[i,j]:>4.0f}/{x0c[i,j]:>4.0f}/{x0s[i,j]:>4.0f})")
    print("-" * 88)

    with open(OUTPUT.replace(".png", "_stations.csv"), "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,scenario_total_mm,scenario_conv_mm,"
                "scenario_lowcloud_mm,conv_share,raw_total,raw_conv,raw_lowcloud\n")
        for (name, lo, la, vt, vc, vs, share) in rows:
            i = int(np.abs(lats - la).argmin()); j = int(np.abs(lons - lo).argmin())
            f.write(f"{name},{lo:.2f},{la:.2f},{vt:.1f},{vc:.1f},{vs:.1f},{share:.2f},"
                    f"{float(x0t[i,j]):.1f},{float(x0c[i,j]):.1f},{float(x0s[i,j]):.1f}\n")
    print("✅ جدول المحطات حُفظ")


if __name__ == "__main__":
    main()
