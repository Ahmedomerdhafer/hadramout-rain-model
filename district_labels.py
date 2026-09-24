# -*- coding: utf-8 -*-
"""
===============================================================================
  أسماء مديريات محافظة حضرموت على الخرائط — عربية (المصدر: GADM v4.1)
===============================================================================
  يرسم اسم كل مديرية عند نقطة تمثيلية داخل حدودها مع هالة بيضاء للوضوح
  فوق الحقول الملونة. الأسماء العربية وفق التقسيم الإداري الرسمي
  (المركز الوطني للمعلومات — اليمن).
===============================================================================
"""
import json
import os
import unicodedata

import matplotlib.patheffects as pe
import cartopy.crs as ccrs
from shapely.geometry import shape, Point

GADM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "gadm41_YEM_2.json")

# الاسم العربي لكل مديرية (مفاتيح GADM NAME_2 حرفياً)
AR_NAMES = {
    "AdDis": "الديس",
    "AdhDhlia'ah": "الضليعة",
    "AlAbr": "العبر",
    "AlMukalla": "المكلا",
    "AlMukallaCity": "مدينة المكلا",
    "AlQaf": "القف",
    "AlQatn": "القطن",
    "Amd": "عمد",
    "ArRaydahWaQusayar": "ريدة وقصيعر",
    "AsSawm": "السوم",
    "AshShihr": "الشحر",
    "BromMayfa": "بروم ميفع",
    "Daw'an": "دوعن",
    "GhaylBaWazir": "غيل باوزير",
    "GhaylBinYamin": "غيل بن يمين",
    "HagrAsSai'ar": "حجر الصيعر",
    "Hajr": "حجر",
    "Hidaybu": "حديبو",
    "Huraidhah": "حريضة",
    "QulensyaWaAbdAlKuri": "قلنسية وعبد الكوري",
    "Rakhyah": "رخية",
    "Rumah": "رماه",
    "Sah": "ساه",
    "Sayun": "سيئون",
    "Shibam": "شبام",
    "Tarim": "تريم",
    "Thamud": "ثمود",
    "WadiAlAyn": "وادي العين",
    "Yabuth": "يبعث",
    "ZamakhwaManwakh": "زمخ ومنوخ",
}

# مراسٍ بلدية معروفة حيث تبتعد نقطة التمثيل الهندسية عن موضع المدينة
# الذي يتوقعه القارئ (تُتحقق الاحتوائية برمجياً وإلا يُستخدم نقطة التمثيل)
ANCHORS = {
    "AlMukalla": (49.283, 14.687),   # المديرية الريفية — أقرب نقطة داخلية للمدينة
    "AshShihr": (49.62, 14.78),      # مدينة الشحر الساحلية
    "GhaylBaWazir": (49.12, 14.86),  # بلدة غيل باوزير
    "Sayun": (48.78, 15.94),         # مدينة سيئون
    "Tarim": (49.00, 16.06),         # مدينة تريم
    "Shibam": (48.62, 15.91),        # مدينة شبام
    "AlQatn": (48.35, 15.90),        # بلدة القطن
}


def _norm(s):
    return unicodedata.normalize("NFKD", s or "").encode(
        "ascii", "ignore").decode()


def add_district_labels(ax, ar, fontfamily, fontsize=8.0, color="#1a1a1a",
                        extent=None, zorder=5.5, halo=1.8):
    """اكتب اسم كل مديرية داخل حدودها (مع هالة بيضاء).

    ملاحظة تنفيذية: تُرسم النصوص على مستوى الشكل (fig.text) بإحداثيات
    العرض المحسوبة من ax.transData — الطريقة الوحيدة الموثوقة مع محاور
    كارتوبي (رسم ax.text بتحويل جغرافي عند مواضع معينة لا يُعرض).
    يجب استدعاؤها بعد اكتمال كل بيانات المحاور (قبل الحفظ).
    """
    fig = ax.get_figure()
    # ثبّت حدود العرض أولاً: كارتوبي يعيد تطبيق النطاق عند أول رسم،
    # فيجب قياس المواضع بعد ذلك وإلا انزاحت الأسماء عن مواضعها.
    fig.canvas.draw()
    with open(GADM_PATH, encoding="utf-8") as f:
        gj = json.load(f)

    n = 0
    for ft in gj["features"]:
        p = ft["properties"]
        if "adram" not in _norm(p.get("NAME_1", "")).lower():
            continue
        arname = AR_NAMES.get(p.get("NAME_2"))
        if not arname:
            continue
        geom = shape(ft["geometry"])
        b = geom.bounds
        if extent and (b[0] > extent[1] or b[2] < extent[0] or
                       b[1] > extent[3] or b[3] < extent[2]):
            continue
        # لأشكال الجزر المتعددة: خذ أكبر مضلع (الكتلة الرئيسية للمديرية)
        if geom.geom_type == "MultiPolygon":
            geom_main = max(geom.geoms, key=lambda g: g.area)
        else:
            geom_main = geom
        pt = geom_main.representative_point()
        # مرساة بلدية معروفة إن وُجدت وداخل حدود المديرية
        anchor = ANCHORS.get(p.get("NAME_2"))
        if anchor and geom.contains(Point(anchor)):
            pt = Point(anchor)
        if extent and not (extent[0] <= pt.x <= extent[1] and
                           extent[2] <= pt.y <= extent[3]):
            continue
        # الموضع الجغرافي → إحداثيات عرض → نسبة الشكل
        # (إحداثيات matplotlib تُقاس من الأسفل — لا انعكاس)
        try:
            disp = ax.transData.transform((pt.x, pt.y))
        except Exception:
            continue
        W, H = fig.get_size_inches() * fig.dpi
        fig.text(disp[0] / W, disp[1] / H, ar(arname),
                 fontsize=fontsize, color=color, ha="center", va="center",
                 fontfamily=fontfamily,
                 path_effects=[pe.withStroke(linewidth=halo,
                                             foreground="white")])
        n += 1
    return n
