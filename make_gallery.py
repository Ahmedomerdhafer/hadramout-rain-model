# -*- coding: utf-8 -*-
"""
===============================================================================
  مولّد صفحة المعاينة الذاتية — كل المنتجات داخل صفحة واحدة بلا أي روابط خارجية
  (المنصة تحمي المعاينة بتوكن وصول → أي تنقل خارج الصفحة يُرفض 403)
  الاستخدام: python3 make_gallery.py   → يعيد توليد index.html
===============================================================================
"""
import csv
import glob
import os

AR_HEADERS = {
    "station": "المحطة", "longitude": "الطول", "latitude": "العرض",
    "scenario_total_24h_mm": "الإجمالي 24س (ملم)", "raw_total_24h_mm": "الخام 24س (ملم)",
    "scenario_total_mm": "الإجمالي (ملم)", "scenario_conv_mm": "الحملي (ملم)",
    "scenario_lowcloud_mm": "الطبقي (ملم)", "conv_share": "الحصة الحملية",
    "raw_total": "خام كلي", "raw_conv": "خام حملي", "raw_lowcloud": "خام طبقي",
}

def latest(pattern, exclude=None):
    fs = sorted(f for f in glob.glob(pattern) if not exclude or exclude not in f)
    if not fs:
        raise SystemExit(f"✗ لا يوجد: {pattern}")
    return fs[-1]

def table_html(csv_path):
    if not os.path.exists(csv_path):
        return "<div style='color:#c0392b;font-size:12px;'>جدول غير موجود</div>"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = [AR_HEADERS.get(h, h) for h in rows[0]]
    out = ["<table style='border-collapse:collapse;width:100%;font-size:12.5px;margin-top:10px;'>",
           "<tr style='background:#0e1929;color:#9fc3e8;'>" +
           "".join(f"<th style='border:1px solid #2c3e57;padding:5px 8px;'>{h}</th>" for h in head) +
           "</tr>"]
    for r in rows[1:]:
        cells = "".join(
            f"<td style='border:1px solid #24324a;padding:4px 8px;text-align:center;color:#dce6f2;'>{c}</td>"
            for c in r)
        out.append(f"<tr style='background:#141d2e;'>{cells}</tr>")
    out.append("</table>")
    return "".join(out)

def section(num, title, png, note="", csv_suffix="_stations.csv"):
    csvp = png.replace(".png", csv_suffix)
    return f"""
  <div style="background:#182234;border:1px solid #2c3e57;border-radius:12px;padding:16px;margin:22px 0;">
    <div style="font-size:18px;font-weight:700;color:#fff;">{num} {title}</div>
    {f"<div style='font-size:12.5px;color:#9fb4cc;margin-top:4px;'>{note}</div>" if note else ""}
    <img src="{png}" alt="{title}" onclick="zoom(this.src)"
         style="width:100%;border-radius:8px;margin-top:12px;display:block;cursor:zoom-in;">
    <div style="font-size:13px;color:#8fd694;margin-top:10px;">📊 جدول المحطات:</div>
    {table_html(csvp)}
  </div>"""

import re
from datetime import datetime, timedelta

MONTHS = {"01": "يناير", "02": "فبراير", "03": "مارس", "04": "أبريل", "05": "مايو",
          "06": "يونيو", "07": "يوليو", "08": "أغسطس", "09": "سبتمبر", "10": "أكتوبر",
          "11": "نوفمبر", "12": "ديسمبر"}

def cyc_of(png):
    m = re.search(r"_(\d{8})_(\d{2})Z", png)
    return f"دورة {int(m.group(1)[6:])} {MONTHS[m.group(1)[4:6]]} / {m.group(2)}Z"


def window_note(png, days):
    """ملاحظة النافذة الزمنية مشتقة من اسم الملف (لا تواريخ مثبتة)"""
    m = re.search(r"_(\d{8})_(\d{2})Z", png)
    if not m:
        return ""
    d0 = datetime.strptime(m.group(1), "%Y%m%d")
    d1 = d0 + timedelta(days=days)
    cc = m.group(2)
    a0 = f"{d0.day} {MONTHS[f'{d0.month:02}']}"
    a1 = f"{d1.day} {MONTHS[f'{d1.month:02}']}"
    if days == 1:
        return f"النافذة: {a0} {cc}:00 ← {a1} {cc}:00 (توقيت غرينتش)"
    return f"النافذة: {a0} ← {a1} {d1.year} ({days} أيام)"

main_png = latest("hadramout_v17_total_24h_*.png")
p24_png = latest("hadramout_v17_3comp_24h_*_panel.png")
p10_png = latest("hadramout_v17_3comp_*_panel.png", exclude="_24h_")
c24, c10 = cyc_of(main_png), cyc_of(p10_png)
as_fs = sorted(glob.glob("hadramout_v17_arabian_sea_14d_*.png"))
as_png = as_fs[-1] if as_fs else None
ai_main_fs = sorted(glob.glob("hadramout_ai_total_24h_*.png"))
ai_main = ai_main_fs[-1] if ai_main_fs else None
ai_p24_fs = sorted(glob.glob("hadramout_ai_3comp_24h_*_panel.png"))
ai_p24 = ai_p24_fs[-1] if ai_p24_fs else None
ai_p10_fs = sorted([f for f in glob.glob("hadramout_ai_3comp_*_panel.png") if "_24h_" not in f])
ai_p10 = ai_p10_fs[-1] if ai_p10_fs else None
ai_as_fs = sorted(glob.glob("hadramout_ai_arabian_sea_14d_*.png"))
ai_as = ai_as_fs[-1] if ai_as_fs else None
cai = f" — خط AI-GFS: {cyc_of(ai_main)}" if ai_main else ""

page = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>خرائط أمطار حضرموت — معاينة</title>
</head>
<body style="margin:0;background:#10141c;color:#e8e8e8;font-family:'Segoe UI',Tahoma,Arial,sans-serif;">

<div id="lb" onclick="this.style.display='none'"
     style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:99;overflow:auto;cursor:zoom-out;text-align:center;">
  <img id="lbi" src="" alt="" style="max-width:none;margin:14px auto;">
  <div style="position:fixed;top:10px;left:14px;color:#fff;font-size:14px;background:#000a;padding:4px 10px;border-radius:6px;">✕ اضغط للإغلاق</div>
</div>
<script>
function zoom(src){{document.getElementById('lbi').src=src;document.getElementById('lb').style.display='block';}}
</script>

<div style="max-width:1450px;margin:0 auto;padding:24px 18px 60px;">

  <div style="text-align:center;padding:22px 12px;background:linear-gradient(135deg,#16202e,#1c2a3d);border-radius:14px;border:1px solid #2c3e57;">
    <div style="font-size:26px;font-weight:800;color:#ffffff;">🌧️ خرائط أمطار محافظة حضرموت</div>
    <div style="font-size:14px;color:#9fb4cc;margin-top:8px;">
      نمذجة سيناريوهية وفق دوال النظام التجريبي NOAA GFS v17-HR1
      <br>منتجات 24 ساعة: {c24} &nbsp;•&nbsp; اللوحة العشرية: {c10} (أحدث دورة مكتملة){cai}
    </div>
    <div style="font-size:12px;color:#7a8fa7;margin-top:6px;">تصميم: أحمد عمر ظافر</div>
  </div>

  <div style="background:#182234;border:1px solid #2c3e57;border-radius:10px;padding:12px 16px;margin:18px 0;font-size:13px;color:#c8d6e8;">
    🔑 المفتاح: أرقام الفئات (ملم) فوق المربعات — من 0.1 إلى 140+ — بألوان مطابقة للنموذج المرجعي المرفق.
    اضغط أي صورة لتكبيرها بالحجم الكامل.
  </div>
{section("①", "الخريطة الرئيسية — إجمالي الأمطار خلال الـ24 ساعة القادمة", main_png,
         window_note(main_png, 1))}
{section("②", "اللوحة الثلاثية — 24 ساعة", p24_png,
         "الإجمالي / الأمطار الرعدية (حملي) / أمطار السحب المنخفضة (طبقي)")}
{section("③", "اللوحة الثلاثية — 10 أيام", p10_png,
         window_note(p10_png, 10) + " — تراكم كلي / حملي / طبقي")}
{section("④", "بحر العرب — التوقعات المدارية خلال 14 يوماً", as_png,
         "التراكم المطري وأقصى الرياح والضغط الأدنى مع مسارات المنخفضات المكتشفة — استرشادي وليس تحذيراً رسمياً",
         csv_suffix="_cities.csv") if as_png else ""}
{section("⑤", "AI-GFS — إجمالي الأمطار خلال الـ24 ساعة القادمة", ai_main,
         "النموذج الذكي NOAA AI-GFS (GraphCast) مُعالَج بمنهج v17-HR1 — قارن مع القسم ①") if ai_main else ""}
{section("⑥", "AI-GFS — اللوحة الثلاثية 24 ساعة", ai_p24,
         "كلي / حملي / طبقي بنفس المنهج — الحصة الحملية من مناخ v17 المرجعي") if ai_p24 else ""}
{section("⑦", "AI-GFS — اللوحة الثلاثية 10 أيام", ai_p10,
         window_note(ai_p10, 10) + " — مقارنة مباشرة مع القسم ③") if ai_p10 else ""}
{section("⑧", "AI-GFS — بحر العرب، التوقعات المدارية 14 يوماً", ai_as,
         "النموذج الذكي GraphCast: مطر ورياح وضغط مع مسارات المنخفضات — قارن مع القسم ④",
         csv_suffix="_cities.csv") if ai_as else ""}

  <div style="text-align:center;font-size:11.5px;color:#66798f;padding:10px;">
    منتجات تجريبية للتوجيه العام — ليست تشغيلاً ديناميكياً لنموذج v17 |
    تُحدَّث المعاينة مع كل دورة بيانات جديدة |
    تصميم: أحمد عمر ظافر
  </div>

</div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(page)
print(f"✅ index.html مُجدّد: {main_png} + {p24_png} + {p10_png}")

# ===== نسخة مضمّنة الصور (preview.html) للعارض المعزول عن الشبكة =====
import base64

def _data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

emb = page
for png in filter(None, (main_png, p24_png, p10_png, as_png, ai_main, ai_p24, ai_p10, ai_as)):
    emb = emb.replace(f'src="{png}"', f'src="{_data_uri(png)}"')
with open("preview.html", "w", encoding="utf-8") as f:
    f.write(emb)
print(f"✅ preview.html (صور مضمّنة base64): {os.path.getsize('preview.html')/1e6:.1f} ميغابايت")
