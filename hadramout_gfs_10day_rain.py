# -*- coding: utf-8 -*-
"""
===============================================================================
  خريطة تراكم التساقط المطري (10 أيام) — محافظة حضرموت، اليمن
  Hadramout Governorate — 10-Day Total Rain Accumulation
===============================================================================
  المصدر       : NOAA GFS v16.3 التشغيلي (FV3-C768 Dynamical Core) — شبكة 0.25°
                 (تم التحقق: المركز KWBC/NCEP، معرف العملية 96 = GFS حسب
                 جدول ON388 Table A، الإصدار التشغيلي v16.3 من مسار NOMADS
                 الرسمي؛ GFS v17 مقرر التنفيذ ~أكتوبر 2026 وفق PNS 26-29)
  المتغير      : APCP (Total Precipitation @ surface) — kg m**-2 == mm
  ملاحظة مهمة  : رسالة APCP في ملفات GFS ذات stepRange يبدأ من الصفر
                 (مثل 0-240) تمثل التراكم الكلي منذ بداية التوقع،
                 لذا فإن الرسالة F240 تساوي مجموع تساقط 10 أيام كاملة.
  التنزيل      : خادم NOMADS (filter_gfs_0p25.pl) مع بديل احتياطي عبر
                 AWS Open Data (noaa-gfs-bdp-pds) بتنزيل جزئي بالبايتات.
  القراءة      : eccodes (GRIB2) مباشرة دون الحاجة إلى xarray/cfgrib.
  الرسم        : matplotlib + cartopy
  العرض        : استيفاء ثنائي الخطية إلى شبكة ≈9 كم (بيانات النموذج 0.25°)
  اللغة        : العنوان بالعربية — يتطلب arabic-reshaper و python-bidi
                 (matplotlib لا يدعم تشكيل الحروف العربية واتجاهها تلقائياً)
                 + خط أميري (Amiri) في مجلد fonts/ أو مثبت في النظام

  الاستخدام:
      python3 hadramout_gfs_10day_rain.py [YYYYMMDD] [Cycle] [عدد الأيام]
      مثال:  python3 hadramout_gfs_10day_rain.py 20260911 00 10

  المتطلبات:
      pip install numpy matplotlib cartopy eccodes ecmwflibs requests \
                  arabic-reshaper python-bidi
===============================================================================
"""

import os
import sys
import unicodedata
import datetime

import numpy as np
import requests
import eccodes as ec

import matplotlib
matplotlib.use("Agg")                     # بيئة بلا شاشة
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ============================ الإعدادات العامة ============================

GFS_RES_DEG = 0.25                        # دقة الشبكة

# نطاق قصّ البيانات عند التنزيل (أوسع قليلاً من حدود الخريطة)
LON_MIN, LON_MAX = 44.0, 56.0
LAT_MIN, LAT_MAX = 11.0, 21.0

# حدود الخريطة النهائية (غرب، شرق، جنوب، شمال)
MAP_EXTENT = [46.2, 53.2, 12.8, 19.5]

RAIN_MIN_MM = 0.1                         # أقل قيمة تُعتبر مطراً (تحتها شفافة)

# دقة شبكة العرض: 9 كم عبر استيفاء ثنائي الخطية من شبكة النموذج 0.25°
# (GFS يُوزَّع رسمياً على 0.25° فقط؛ الاستيفاء ينعّم الحقول ويدقّق حدود
#  الفئات اللونية لكنه لا يضيف معلومات ديناميكية جديدة)
DISPLAY_RES_KM = 9.0
DISPLAY_RES_DEG = DISPLAY_RES_KM / 111.32     # ≈ 0.0809 درجة

CACHE_DIR = "gfs_cache"

NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
AWS_BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Hadramout-GFS-10day-rain/1.0"
}

# معلومات الدورة التي تم استخدامها فعلياً (تتشاركها دالتا التنزيل والرسم)
META = {"date_str": None, "cycle": None, "fxx": None, "source": None,
        "gfs_version": None, "gfs_core": None, "gfs_native_km": None}

# ========================== دوال مساعدة للتنزيل ==========================


def _probe_cycle(date_str, cycle):
    """التحقق من توفر دورة GFS معينة عبر فحص ملف الفهرس F000 على AWS."""
    url = f"{AWS_BUCKET}/gfs.{date_str}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f000.idx"
    try:
        r = requests.head(url, headers=HTTP_HEADERS, timeout=30, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


# خواص إصدارات GFS المعروفة: النواة الديناميكية والدقة الأصلية بالكيلومتر
GFS_VERSION_INFO = {
    "v16.3": ("FV3-C768", 13),
    "v16.4": ("FV3-C768", 13),
    "v17.0": ("FV3-C1152", 9),
    "v17.1": ("FV3-C1152", 9),
    "v17.2": ("FV3-C1152", 9),
    "v17.3": ("FV3-C1152", 9),
}


def _detect_gfs_version(date_str, cycle):
    """اكتشاف إصدار GFS التشغيلي الفعلي من مسارات NOMADS المُصدَّرة.

    يفحص المسارات الرسمية (com/gfs/vX.Y/...) ويعيد أحدث إصدار يحتوي
    الدورة المطلوبة. عند تنفيذ GFS v17 رسمياً (المتوقع ~أكتوبر 2026 وفق
    PNS 26-29) سيتحول السكربت إليه تلقائياً — البيانات ستتدفق عبر نفس
    القنوات (NOMADS/AWS) وملفات 0.25° pgrb2 ستبقى متاحة (PNS 26-30).
    ملاحظة: النسخة التجريبية من v17 تعمل فعلاً لدى NOAA/EMC للتقييم،
    لكن مخرجات GRIB2 الخاصة بها لا تُنشر علناً قبل التنفيذ الرسمي.
    """
    for v in ["v17.3", "v17.2", "v17.1", "v17.0", "v16.4", "v16.3"]:
        url = (f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/{v}/"
               f"gfs.{date_str}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f000.idx")
        try:
            r = requests.get(url, headers={"Range": "bytes=0-99", **HTTP_HEADERS},
                             timeout=20)
            if r.status_code in (200, 206):
                return v
        except requests.RequestException:
            pass
    return None


def _build_candidates(date_str, cycle):
    """قائمة الدورات المرشحة: المطلوبة أولاً ثم الأحدث المتاح للأيام السابقة."""
    cands = [(date_str, cycle)]
    base = datetime.datetime.strptime(date_str, "%Y%m%d")
    for dd in range(1, 8):
        d = (base - datetime.timedelta(days=dd)).strftime("%Y%m%d")
        for c in ("00", "12", "06", "18"):
            cands.append((d, c))
    return cands


def _fxx_candidates(target_hours):
    """ساعات التوقع المرشحة (بعد F120 تكون المخرجات كل 3 ساعات)."""
    return [target_hours - 3 * i for i in range(5)]


def _fetch_nomads(date_str, cycle, fxx, dest):
    """تنزيل متغير APCP (سطح) فقط من NOMADS عبر خدمة التصفية."""
    params = {
        "file": f"gfs.t{cycle}z.pgrb2.0p25.f{fxx:03d}",
        "lev_surface": "on",
        "var_APCP": "on",
        "dir": f"/gfs.{date_str}/{cycle}/atmos",
    }
    for attempt in (1, 2):
        try:
            r = requests.get(NOMADS_FILTER, params=params,
                             headers=HTTP_HEADERS, timeout=300)
            if r.status_code == 200 and r.content[:4] == b"GRIB":
                with open(dest, "wb") as f:
                    f.write(r.content)
                return dest
        except requests.RequestException as e:
            print(f"   ⚠️ محاولة NOMADS {attempt} فشلت: {e}")
    return None


def _fetch_aws_range(date_str, cycle, fxx, dest):
    """بديل احتياطي: تنزيل رسالة APCP التراكمية فقط من AWS عبر HTTP Range."""
    base = (f"{AWS_BUCKET}/gfs.{date_str}/{cycle}/atmos/"
            f"gfs.t{cycle}z.pgrb2.0p25.f{fxx:03d}")
    try:
        idx = requests.get(base + ".idx", headers=HTTP_HEADERS, timeout=60)
        if idx.status_code != 200:
            return None
        entries = []
        for line in idx.text.splitlines():
            parts = line.split(":")
            if len(parts) >= 6:
                try:
                    entries.append((int(parts[1]), parts[3], parts[4], parts[5]))
                except ValueError:
                    continue
        # نختار رسالة APCP السطحية ذات النافذة التراكمية من الصفر (0-...)
        chosen = None
        for i, (s, var, lev, desc) in enumerate(entries):
            if var == "APCP" and lev == "surface" and desc.strip().startswith("0-"):
                chosen = i
        if chosen is None:
            return None
        start = entries[chosen][0]
        if chosen + 1 < len(entries):
            end = entries[chosen + 1][0] - 1
        else:
            cl = requests.head(base, headers=HTTP_HEADERS, timeout=60)
            end = int(cl.headers.get("Content-Length", 0)) - 1
        r = requests.get(base, headers={"Range": f"bytes={start}-{end}", **HTTP_HEADERS},
                         timeout=180)
        if r.status_code in (200, 206) and r.content[:4] == b"GRIB":
            with open(dest, "wb") as f:
                f.write(r.content)
            return dest
    except requests.RequestException as e:
        print(f"   ⚠️ فشل التنزيل من AWS: {e}")
    return None


# ============================ قراءة GRIB2 ================================


def _read_cumulative_apcp(path, target_hours):
    """قراءة رسالة APCP التراكمية (startStep=0) من ملف GRIB2.

    تعيد (القيم ثنائية الأبعاد، خطوط العرض، خطوط الطول، endStep)
    أو None إن لم توجد رسالة مناسبة.
    """
    best = None
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                if ec.codes_get(gid, "shortName") != "tp":
                    continue
                s = int(ec.codes_get(gid, "startStep"))
                e = int(ec.codes_get(gid, "endStep"))
                # نريد فقط التراكم منذ بداية التوقع (0 -> Fxxx)
                if s != 0 or e <= 0 or e > target_hours:
                    continue
                if best is not None and e <= best[0]:
                    continue
                ni = int(ec.codes_get(gid, "Ni"))
                nj = int(ec.codes_get(gid, "Nj"))
                lat1 = float(ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
                lat2 = float(ec.codes_get(gid, "latitudeOfLastGridPointInDegrees"))
                lon1 = float(ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
                lon2 = float(ec.codes_get(gid, "longitudeOfLastGridPointInDegrees"))
                glats = lat1 + (lat2 - lat1) / (nj - 1) * np.arange(nj)
                glons = lon1 + (lon2 - lon1) / (ni - 1) * np.arange(ni)
                vals = ec.codes_get_values(gid).reshape(nj, ni)
                best = (e, vals, glats, glons)
            except ec.KeyError:
                continue
            finally:
                ec.codes_release(gid)
    if best is None:
        return None
    e, vals, glats, glons = best
    return vals, glats, glons, e


def _subset_region(vals, glats, glons):
    """قصّ نطاق حضرموت من الشبكة مع ضمان ترتيب الإحداثيات تصاعدياً."""
    glons360 = np.where(glons < 0, glons + 360.0, glons)
    mask_lon = (glons360 >= LON_MIN) & (glons360 <= LON_MAX)
    mask_lat = (glats >= LAT_MIN) & (glats <= LAT_MAX)
    sub = vals[np.ix_(mask_lat, mask_lon)]
    lats_s = glats[mask_lat]
    lons_s = glons360[mask_lon]
    if lats_s.size and lats_s[0] > lats_s[-1]:
        lats_s = lats_s[::-1]
        sub = sub[::-1, :]
    return sub, lats_s, lons_s


# ==================== الدالة الرئيسية للتنزيل والتراكم ====================


def download_and_accumulate_gfs_precip(date_str, cycle="00", total_days=10):
    """تنزيل بيانات GFS وحساب التساقط المتراكم لكامل فترة التوقع.

    بما أن رسالة APCP ذات النافذة (0-Fxxx) تراكمية منذ بداية التوقع،
    فإن تنزيل الإطار الأخير (F240 لعشرة أيام) يعطي المجموع الكلي مباشرة.
    تعيد (precip_mm, lats, lons) أو (None, None, None) عند الفشل.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    target_hours = int(total_days) * 24

    print("=" * 74)
    print(f"🌍 تنزيل GFS 0.25° — التاريخ: {date_str} | الدورة: {cycle}Z | "
          f"فترة التراكم: {total_days} أيام ({target_hours} ساعة)")
    print("=" * 74)

    # اكتشاف إصدار GFS التشغيلي (يتحول تلقائياً إلى v17 عند تنفيذه)
    gfs_version = _detect_gfs_version(date_str, cycle)
    if gfs_version:
        core, native_km = GFS_VERSION_INFO.get(gfs_version, ("FV3", 13))
        META.update(gfs_version=gfs_version, gfs_core=core, gfs_native_km=native_km)
        print(f"🔎 الإصدار التشغيلي المكتشف: GFS {gfs_version} "
              f"(النواة {core} — دقة أصلية ≈{native_km} كم)")
        if gfs_version.startswith("v17"):
            print("🚀 GFS v17 أصبح تشغيلياً — سيُستخدم في هذه الخريطة!")
        else:
            print("ℹ️ v17 التجريبي قيد التقييم لدى NOAA/EMC ولا تُنشر مخرجاته "
                  "علناً بعد؛ سيُكتشف تلقائياً فور التنفيذ الرسمي (~أكتوبر 2026)")
    else:
        print("⚠️ تعذر تحديد رقم الإصدار من مسارات NOMADS")

    for d, c in _build_candidates(date_str, cycle):
        if not _probe_cycle(d, c):
            continue
        if (d, c) != (date_str, cycle):
            print(f"⚠️ الدورة المطلوبة غير متاحة — سيتم استخدام: {d} / {c}Z")

        for fxx in _fxx_candidates(target_hours):
            cache_path = os.path.join(CACHE_DIR, f"gfs.{d}-{c}z.f{fxx:03d}.apcp.grib2")
            if os.path.exists(cache_path):
                source = "NOMADS/AWS (ذاكرة مؤقتة محلية)"
            else:
                src = _fetch_nomads(d, c, fxx, cache_path)
                source = "NOMADS (filter_gfs_0p25)"
                if src is None:
                    src = _fetch_aws_range(d, c, fxx, cache_path)
                    source = "AWS Open Data (noaa-gfs-bdp-pds)"
                if src is None:
                    continue

            res = _read_cumulative_apcp(cache_path, target_hours)
            if res is None:
                continue
            vals, glats, glons, end_step = res
            precip, lats, lons = _subset_region(vals, glats, glons)

            if end_step < target_hours:
                print(f"⚠️ آخر إطار تراكمي متاح: F{end_step:03d} بدلاً من F{target_hours:03d}")

            print(f"📥 تم التنزيل والقراءة بنجاح — المصدر: {source}")
            print(f"   الرسالة: APCP سطح | نافذة التراكم: 0–{end_step} ساعة | الوحدة: ملم (kg m⁻²)")
            print(f"   نطاق الشبكة: lon {lons.min():.2f}→{lons.max():.2f} | "
                  f"lat {lats.min():.2f}→{lats.max():.2f} | الأبعاد: {len(lats)}×{len(lons)}")
            print(f"   إحصاءات النطاق: الأعلى = {precip.max():.1f} ملم | "
                  f"المتوسط = {precip.mean():.2f} ملم")

            META.update(date_str=d, cycle=c, fxx=end_step, source=source)
            return precip, lats, lons

    print("❌ تعذر تنزيل بيانات GFS من جميع المصادر المرشحة.")
    return None, None, None


# ============================== أدوات الرسم ==============================


def _ar(text):
    """تهيئة النص العربي للعرض الصحيح داخل matplotlib.

    matplotlib لا يقوم بتشكيل الحروف العربية (ربطها بصيغها المتصلة)
    ولا بترتيب الاتجاه من اليمين إلى اليسار، لذا نستخدم:
      - arabic_reshaper : تحويل الحروف إلى Arabic Presentation Forms المتصلة
      - python-bidi     : إعادة ترتيب النص بصرياً حسب خوارزمية الاتجاه
    تُطبَّق سطراً سطراً لدعم النصوص متعددة الأسطر.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        print("⚠️ مكتبتا arabic-reshaper و python-bidi غير مثبتتين — "
              "سيظهر النص العربي بشكل غير صحيح!")
        return text
    return "\n".join(get_display(arabic_reshaper.reshape(line))
                     for line in text.split("\n"))


def _setup_arabic_font():
    """تسجيل خط عربي مناسب وإعادة قائمة عائلات خطوط للنص العربي.

    يُفضَّل خط أميري (Amiri) لأنه يغطي كامل أشكال العرض العربية
    (المتصلة والمنعزلة) التي ينتجها arabic_reshaper — معظم الخطوط
    الحديثة (Tajawal/Noto/…) لا تتضمن هذه الأشكال لأنها تعتمد على
    تشكيل OpenType الذي لا يدعمه matplotlib.
    ملاحظة مهمة: تمرير قائمة خطوط صريحة (وليس الاسم المستعار
    sans-serif) هو ما يُفعِّل التراجع التدريجي بين الخطوط glyph-by-glyph،
    فتُؤخذ الحروف العربية من أميري وأي رمز ناقص (مثل ←) من DejaVu.
    """
    from matplotlib import font_manager
    families = []
    # أولاً: ملفات الخطوط في مجلد fonts/ بجوار السكربت
    fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if os.path.isdir(fdir):
        for fn in sorted(os.listdir(fdir)):
            if fn.lower().endswith((".ttf", ".otf")):
                try:
                    fp = os.path.join(fdir, fn)
                    font_manager.fontManager.addfont(fp)
                    fam = font_manager.FontProperties(fname=fp).get_name()
                    if fam not in families:
                        families.append(fam)
                except Exception:
                    pass
    # ثانياً: خطوط عربية قد تكون مثبتة في النظام
    preferred = ["Amiri", "Scheherazade New", "Noto Naskh Arabic",
                 "Noto Sans Arabic", "Almarai", "Tajawal"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families += [p for p in preferred if p in installed and p not in families]
    # خط احتياطي نهائي لكل رمز ناقص — DejaVu Sans يغطي العربية كاملة أيضاً
    if "DejaVu Sans" not in families:
        families.append("DejaVu Sans")
    if families:
        print(f"🔤 خط العنوان العربي: {families[0]} "
              f"(احتياط: {', '.join(families[1:])})")
    return families


def _interp_to_fine_grid(precip, lats, lons, res_deg):
    """استيفاء ثنائي الخطية لشبكة أدق للعرض (مثل 9 كم بدل 0.25°).

    يعيد (القيم المستوفاة، خطوط العرض الجديدة، خطوط الطول الجديدة).
    لا يضيف الاستيفاء معلومات جديدة — يمنح مظهراً أنعم وحدوداً أوضح فقط.
    """
    nlat = int(round((lats[-1] - lats[0]) / res_deg)) + 1
    nlon = int(round((lons[-1] - lons[0]) / res_deg)) + 1
    new_lats = np.linspace(lats[0], lats[-1], nlat)
    new_lons = np.linspace(lons[0], lons[-1], nlon)

    # مواقع كل عقدة جديدة في فضاء فهارس الشبكة الأصلية (قيم كسرية)
    lat_f = np.interp(new_lats, lats, np.arange(len(lats)))
    lon_f = np.interp(new_lons, lons, np.arange(len(lons)))
    y0 = np.floor(lat_f).astype(int)
    x0 = np.floor(lon_f).astype(int)
    y1 = np.minimum(y0 + 1, len(lats) - 1)
    x1 = np.minimum(x0 + 1, len(lons) - 1)
    wy = (lat_f - y0)[:, None]      # الوزن بين الصف السفلي والعلوي
    wx = (lon_f - x0)[None, :]      # الوزن بين العمود الأيسر والأيمن

    p00 = precip[np.ix_(y0, x0)]
    p01 = precip[np.ix_(y0, x1)]
    p10 = precip[np.ix_(y1, x0)]
    p11 = precip[np.ix_(y1, x1)]
    top = p00 * (1 - wx) + p01 * wx
    bot = p10 * (1 - wx) + p11 * wx
    fine = top * (1 - wy) + bot * wy
    return np.maximum(fine, 0.0), new_lats, new_lons


def _add_yemen_admin1(ax):
    """رسم حدود المحافظات اليمنية مع تمييز حدود حضرموت بحد أوضح."""
    from cartopy.io.shapereader import natural_earth, Reader
    from cartopy.feature import ShapelyFeature

    shp_path = natural_earth(resolution="10m", category="cultural",
                             name="admin_1_states_provinces")
    yemen_geoms, hadramout_geom = [], None
    for rec in Reader(shp_path).records():
        attrs = rec.attributes
        if attrs.get("admin") != "Yemen":
            continue
        yemen_geoms.append(rec.geometry)
        name = unicodedata.normalize("NFKD", attrs.get("name") or "").encode(
            "ascii", "ignore").decode()
        if "adram" in name.lower().replace("-", "").replace(" ", ""):
            hadramout_geom = rec.geometry

    if yemen_geoms:
        ax.add_feature(ShapelyFeature(yemen_geoms, ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="#a29a8e",
                                      linewidth=0.45, zorder=3))
    if hadramout_geom is not None:
        ax.add_feature(ShapelyFeature([hadramout_geom], ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="#6d4c2a",
                                      linewidth=2.0, zorder=4))
    return hadramout_geom is not None


def _add_hadramout_districts(ax):
    """رسم حدود مديريات محافظة حضرموت (المصدر: قاعدة GADM v4.1).

    ينزّل ملف حدود اليمن الإدارية (مستوى المديريات ADM2) تلقائياً عند أول
    تشغيل ويحفظه في مجلد data/، ثم يرسم حدود مديريات حضرموت الثلاثين فقط.
    """
    import json
    import zipfile
    from shapely.geometry import shape
    from cartopy.feature import ShapelyFeature

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    gj_path = os.path.join(data_dir, "gadm41_YEM_2.json")
    if not os.path.exists(gj_path):
        zip_path = gj_path + ".zip"
        url = ("https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
               "gadm41_YEM_2.json.zip")
        print("⬇️ تنزيل حدود المديريات (GADM v4.1)...")
        r = requests.get(url, headers=HTTP_HEADERS, timeout=300)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)

    with open(gj_path, encoding="utf-8") as f:
        gj = json.load(f)

    def _norm(s):
        return unicodedata.normalize("NFKD", s or "").encode(
            "ascii", "ignore").decode()

    geoms = [shape(ft["geometry"]) for ft in gj["features"]
             if "adram" in _norm(ft["properties"].get("NAME_1", "")).lower()]
    if not geoms:
        print("⚠️ لم تُعثر على مديريات حضرموت في ملف الحدود")
        return 0
    ax.add_feature(ShapelyFeature(geoms, ccrs.PlateCarree(),
                                  facecolor="none", edgecolor="#6b5d4a",
                                  linewidth=0.55, zorder=3.5))
    print(f"🏛️ تم رسم حدود {len(geoms)} مديرية من مديريات حضرموت (GADM v4.1)")
    return len(geoms)


# =========================== دالة رسم الخريطة ============================


def plot_10day_accumulated_rain(precip_10d, lats, lons, output_img=None):
    """رسم خريطة تراكم التساقط المطري لعشرة أيام فوق محافظة حضرموت."""

    date_str = META.get("date_str")
    cycle = META.get("cycle") or "00"
    fxx = META.get("fxx") or 240
    source = META.get("source") or "NOAA GFS"

    if output_img is None:
        output_img = f"hadramout_10day_rain_{date_str or 'gfs'}_{cycle}Z.png"

    init_dt = datetime.datetime.strptime(date_str + cycle, "%Y%m%d%H") if date_str \
        else None
    end_dt = init_dt + datetime.timedelta(hours=int(fxx)) if init_dt else None

    # ---------------------- إعداد الشكل والمحاور ----------------------
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
        print(f"⚠️ تعذر رسم حدود المحافظات: {e}")

    # حدود مديريات حضرموت (ADM2)
    try:
        _add_hadramout_districts(ax)
    except Exception as e:
        print(f"⚠️ تعذر رسم حدود المديريات: {e}")

    # ---------------------- طبقة المطر المتراكم ----------------------
    # استيفاء إلى شبكة عرض أدق (~9 كم) لحدود فئات أنعم وأوضح
    precip_fine, lats_f, lons_f = _interp_to_fine_grid(
        precip_10d, lats, lons, DISPLAY_RES_DEG)
    print(f"🗺️ شبكة العرض: {len(lats_f)}×{len(lons_f)} عقدة "
          f"(≈{DISPLAY_RES_KM:.0f} كم — استيفاء ثنائي الخطية من 0.25°)")
    precip_plot = np.where(precip_fine >= RAIN_MIN_MM, precip_fine, np.nan)
    # 20 فئة بألوان ميتيو بلو الرسمية (لوحة رادار meteoblue: Drizzle→Hail)
    # المراسي الست مأخوذة من CSS الموقع الرسمي وموصولة بمشية متساوية القوس في Lab
    # كل فئة متمايزة عن جارتها (ΔE ≥ 15 — نفس فجوة فئات meteoblue الأصلية)
    levels = [0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 140]
    rain_colors = [
        "#7AE1E8",  # 0.1–0.5   رشّ (Drizzle)
        "#0AC5D9",  # 0.5–1     خفيف (Light)
        "#28B2DE",  # 1–2       سيان مزرق
        "#31A0E2",  # 2–3       أزرق فاتح
        "#328DE6",  # 3–4       أزرق
        "#337AEA",  # 4–6       معتدل (Moderate)
        "#736FEA",  # 6–8       بنفسجي مزرق
        "#9860EB",  # 8–10      بنفسجي فاتح
        "#B94AEB",  # 10–12     غزير (Heavy)
        "#CB49D4",  # 12–15     أرجواني
        "#D64DBB",  # 15–20     أرجواني وردي
        "#DF51A0",  # 20–25     وردي أرجواني
        "#E65687",  # 25–30     وردي
        "#EB5B6F",  # 30–40     وردي محمر
        "#EF6152",  # 40–50     برتقالي محمر
        "#F27343",  # 50–60     غزير جداً (Very Heavy)
        "#F69343",  # 60–75     برتقالي
        "#F9AF42",  # 75–90     برتقالي أصفر
        "#FACA40",  # 90–110    أصفر برتقالي
        "#FAE63C",  # 110–140   برد (Hail)
    ]
    cmap = ListedColormap(rain_colors)
    cmap.set_over("#050A3C")     # لون مثلث الامتداد (أكثر من 140 ملم)
    cmap.set_bad(alpha=0.0)      # دون 0.1 ملم = شفاف
    norm = BoundaryNorm(levels, cmap.N)

    dry_everywhere = bool(np.all(np.isnan(precip_plot)))
    cf = None
    if not dry_everywhere:
        cf = ax.contourf(lons_f, lats_f, precip_plot, levels=levels, cmap=cmap,
                         norm=norm, extend="max", alpha=1.0, zorder=2,
                         transform=ccrs.PlateCarree())
        cb = fig.colorbar(cf, ax=ax, orientation="vertical", shrink=0.80,
                          pad=0.015, aspect=32)
        cb.set_label("Total 10-Day Precipitation (mm)", fontsize=10)
        cb.set_ticks(levels)
        cb.ax.tick_params(labelsize=7.5)
    else:
        ax.text(0.5, 0.5, "No significant precipitation expected (< 0.1 mm)",
                transform=ax.transAxes, ha="center", fontsize=13, color="#777777")

    # إحصاءات داخل حدود الخريطة
    mlat = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    mlon = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    sub = precip_10d[np.ix_(mlat, mlon)]
    sub_lats = lats[mlat]
    sub_lons = lons[mlon]
    vmax = float(sub.max())
    vmean = float(sub.mean())
    imax = np.unravel_index(np.nanargmax(sub), sub.shape)
    vmax_lat, vmax_lon = float(sub_lats[imax[0]]), float(sub_lons[imax[1]])

    # ---------------------- النقاط المرجعية الحيوية ----------------------
    stations = {
        'Mukalla': (49.12, 14.53),
        'Ash Shihr': (49.60, 14.76),
        'Al-Dhabba Port': (49.50, 14.70),
        'Shuhayr': (49.42, 14.65),
        'Wadi Huwayrah': (49.35, 14.82),
        'Seiyun (Wadi)': (48.78, 15.93),
        'Tarim': (49.00, 16.05),
        'Dawan (Wadi)': (48.45, 15.00)
    }

    # النقاط المرجعية — علامات فقط دون أسماء نصية (حسب الطلب)
    station_rows = []
    for name, (st_lon, st_lat) in stations.items():
        ax.plot(st_lon, st_lat, marker='o', markersize=4.5, color='darkred',
                transform=ccrs.PlateCarree(), zorder=6)

        # قيمة التراكم عند أقرب نقطة شبكة لكل محطة (من شبكة النموذج الأصلية 0.25°)
        iy = int(np.abs(lats - st_lat).argmin())
        ix = int(np.abs(lons - st_lon).argmin())
        station_rows.append((name, st_lon, st_lat, float(precip_10d[iy, ix])))

    # صندوق إحصاءات داخل الخريطة
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

    # ---------------------- العنوان والحواشي ----------------------
    # العنوان باللغة العربية — يُعالج عبر _ar() لربط الحروف وضبط الاتجاه
    # (محاذاة اليمين هي الوضع الطبيعي للعناوين العربية)
    # fontfamily كقائمة صريحة تُفعِّل التراجع التدريجي بين الخطوط
    # بناء سطر النموذج ديناميكياً حسب الإصدار التشغيلي المكتشف
    gv = META.get("gfs_version") or "v16.3"
    core = META.get("gfs_core") or "FV3-C768"
    if gv.startswith("v17"):
        model_line = (f"نموذج NOAA GFS {gv} التشغيلي "
                      f"(النواة {core} — دقة أصلية ≈9 كم)")
    else:
        model_line = f"نموذج NOAA GFS {gv} التشغيلي (النواة الديناميكية {core})"

    plt.title(_ar("محافظة حضرموت — إجمالي تراكم الأمطار خلال 10 أيام\n"
                  + model_line + " | دقة العرض ≈ 9 كم"),
              fontsize=13, fontweight='bold', loc='right', pad=12,
              fontfamily=arabic_fonts)

    if init_dt and end_dt:
        ax.text(0.015, 0.985,
                f"Init: {init_dt:%d %b %Y %H}Z  |  Valid: {init_dt:%d %b} → {end_dt:%d %b %Y %H}Z  (F000–F{fxx:03d})",
                transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
                color='#222222', fontweight='bold', zorder=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.75, edgecolor='#bbbbbb'))
        fig.text(0.09, 0.004,
                 f"Data: NOAA/NCEP operational GFS {gv}, 0.25° GRIB2 — KWBC, generating process 96 (ON388 Table A) — APCP surface (cumulative F000–F{fxx:03d}) via {source}\n"
                 f"Display grid ~9 km (bilinear interpolation of the 0.25° model grid)  |  Admin boundaries: Natural Earth (governorates) + GADM v4.1 (districts)  |  "
                 f"Station values = nearest 0.25° model grid point\n"
                 f"Note: experimental GFS v17 (FV3-C1152) is running at NOAA/EMC for evaluation; its GRIB2 output is not publicly distributed until operational implementation (~Oct 2026, PNS 26-29) — this map auto-detects and switches to GFS v17 on implementation day",
                 fontsize=7.5, color='#555555', ha='left', va='bottom')

    plt.savefig(output_img, bbox_inches='tight')
    plt.close()
    print(f"✅ تم تصدير خريطة التراكم بنجاح: {output_img}")

    # ---------------------- تقرير المحطات ----------------------
    print("\n📊 التساقط المتراكم المتوقع (10 أيام) عند النقاط المرجعية:")
    print("-" * 64)
    for name, lon, lat, v in station_rows:
        bar = "█" * int(round(min(v, 50) / 5))
        print(f"  {name:<16} ({lat:5.2f}N, {lon:5.2f}E) : {v:7.1f} mm  {bar}")
    print("-" * 64)

    csv_path = output_img.rsplit(".", 1)[0] + "_stations.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("station,longitude,latitude,precip_mm_10day\n")
        for name, lon, lat, v in station_rows:
            f.write(f"{name},{lon:.2f},{lat:.2f},{v:.1f}\n")
    print(f"✅ تم تصدير جدول المحطات: {csv_path}")


# =============================== التنفيذ =================================

if __name__ == "__main__":
    # يمكن إدخال تاريخ اليوم وتشغيل دورة الفجر 00Z
    date_str = sys.argv[1] if len(sys.argv) > 1 else "20260911"
    cycle = sys.argv[2] if len(sys.argv) > 2 else "00"
    total_days = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    precip_10d, lats, lons = download_and_accumulate_gfs_precip(
        date_str=date_str, cycle=cycle, total_days=total_days)

    if precip_10d is not None:
        plot_10day_accumulated_rain(precip_10d, lats, lons)
    else:
        print("❌ فشل إنتاج الخريطة لعدم توفر البيانات.")
