# -*- coding: utf-8 -*-
"""
===============================================================================
  مفتاح خرائط أفقي — مطابق حرفياً للصورة المرفقة (طراز WeatherBELL)
===============================================================================
  البنية (كما في الصورة): شريط مربعات ملونة متجاورة، ورقم حد الفئة الأدنى
  فوق كل مربع. المربع الأخير مفتوح (≥ الحد الأعلى).
  الألوان الـ21 مستخرجة بكسلياً من الصورة المرفقة (طراز WeatherBELL)،
  والفئات مشدودة لمدى قيم منتجات حضرموت (24 ساعة / 10 أيام).
===============================================================================
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# فئات مشدودة لمدى قيم منتجاتنا (بطلب المستخدم): الحد الأدنى لكل مربع؛
# الأخير مفتوح (≥ 140 ملم). الألوان تبقى مطابقة للصورة حرفياً وبالترتيب نفسه.
WX_LEVELS = [0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 140]

# الألوان الـ21 المستخرجة من شريط الصورة مباشرة:
# رمادي ← أخضر فاتح ← أخضر ← أخضر داكن ← أزرق داكن ← أزرق ← أزرق فاتح
# ← أصفر ← برتقالي ← أحمر ← أحمر داكن ← بني ← رملي ← بنفسجي فاتح ← بنفسجي
# ← بنفسجي داكن ← أرجواني
WX_COLORS = [
    "#C0C0C0", "#838383", "#BBF1AF", "#79F374",
    "#1EB420", "#1464CF", "#50A4F3", "#B5F0FB",
    "#FDE979", "#FCA50E", "#FA3603", "#C00402",
    "#870002", "#8C645C", "#C9A098", "#D1CADB",
    "#9A8CBB", "#685393", "#8B0389", "#BD08BB",
    "#D802DB",
]
WX_OVER = "#D802DB"   # ≥ 500 ملم — لون المربع الأخير (المفتوح)


def _fmt(v):
    return ("%g" % v)


def draw_wxbell_legend(fig, x0, x1, y_bar, bar_h=0.022,
                       levels=None, colors=None, over_color=None,
                       unit_ar="ملم", num_fs=8.0, unit_fs=9.0,
                       edge="#4A4A4A", arabic_font=None, ar=None):
    """ارسم المفتاح مطابقاً للصورة: أرقام الفئات فوق المربعات.

    x0, x1, y_bar, bar_h : إحداثيات الشكل (figure fraction) — y_bar أسفل الشريط
    levels : حدود الفئات الدنيا (len(levels) == len(colors))
    returns: محاور المفتاح
    """
    levels = WX_LEVELS if levels is None else levels
    colors = WX_COLORS if colors is None else colors
    over_color = WX_OVER if over_color is None else over_color
    n = len(colors)

    ax = fig.add_axes([x0, y_bar, x1 - x0, bar_h])
    ax.set_xlim(-1.35, n + 1.35)   # هوامش متوازنة → المربعات تتوسّط بين x0 و x1
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=c,
                               edgecolor=edge, linewidth=0.5))

    # رقم الحد الأدنى فوق كل مربع (كما في الصورة المرفقة — المربع الأخير مفتوح)
    for i, lv in enumerate(levels):
        ax.text(i + 0.5, 1.10, _fmt(lv), ha="center", va="bottom",
                fontsize=num_fs, color="#1A1A1A")

    # وحدة القياس يسار صف الأرقام
    if unit_ar:
        kw = {}
        if arabic_font is not None:
            kw["fontfamily"] = arabic_font
        if ar is not None:
            unit_ar = ar(unit_ar)
        ax.text(-0.30, 1.10, unit_ar, ha="right", va="bottom",
                fontsize=unit_fs, color="#1A1A1A", fontweight="bold", **kw)
    return ax
