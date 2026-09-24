#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""توليد hadramout_v17_3comp_panel_24h.py من نسخة الأيام العشرة بتحويلات مُتحقَّقة"""
p_src = "hadramout_v17_3comp_panel.py"
p_dst = "hadramout_v17_3comp_panel_24h.py"
s = open(p_src, encoding="utf-8").read()

reps = [
 # 1) سطر التوثيق الأول
 ("— الأيام العشرة القادمة — محافظة حضرموت",
  "— الـ24 ساعة القادمة — محافظة حضرموت"),
 # 2) ترويسة الدورة/النافذة
 ("الدورة         : 2026-09-14 00Z (أحدث دورة مكتملة عند الإنتاج)\n                   النافذة: 14 سبتمبر 00Z → 24 سبتمبر 00Z",
  "الدورة         : 2026-09-14 06Z (أحدث دورة مكتملة عند الإنتاج)\n                   النافذة: الـ24 ساعة القادمة (14 سبتمبر 06Z → 15 سبتمبر 06Z)"),
 # 3) ملف الإخراج
 ('OUTPUT = "hadramout_v17_3comp_20260914_00Z_panel.png"',
  'OUTPUT = "hadramout_v17_3comp_24h_20260914_06Z_panel.png"'),
 # 4) ملف البيانات
 ('np.load("gfs_cache/v17replay/scenario_3comp_2026091400.npz")',
  'np.load("gfs_cache/v17replay/scenario_3comp_24h_2026091406.npz")'),
 # 5) صندوق النافذة
 ('"Window: 14 Sep 2026 00Z → 24 Sep 2026 00Z (10 days)\\n"\n                 "Cycle: 20260914/00Z — latest complete run",',
  '"Window: next 24 h — 14 Sep 2026 06Z → 15 Sep 2026 06Z\\n"\n                 "Cycle: 20260914/06Z — latest complete run",'),
 # 6) العنوان الرئيسي
 ("خلال الأيام العشرة القادمة", "خلال الـ24 ساعة القادمة"),
 # 7) شريط الألوان
 ('cb.set_label("10-Day Total Precipitation (mm)", fontsize=10)',
  'cb.set_label("24-Hour Total Precipitation (mm)", fontsize=10)'),
 # 8) الحاشية: الدورة
 ("cycle 20260914/00Z ", "cycle 20260914/06Z "),
 # 9) الحاشية: المنهجية
 (('(APCP total & ACPCP convective, cumulative 0–240 h)\\n"',
   '(APCP total & ACPCP convective, cumulative 0–24 h)\\n"\n'
   '             "v17-HR1 climatology scaled to the 24-h window via the reference-sample '
   '24 h/10-day ratio (shape α unchanged)\\n"')),
 # 10) الحاشية: العينة المرجعية
 ("Reference: latest 24 GFS runs (23 Aug–12 Sep 2026)",
  "Reference: the same 24 GFS runs — paired 0–24 h & 0–240 h samples (23 Aug–12 Sep 2026)"),
 # 11) ترويسة تقرير المحطات
 ("(دورة 14/00Z — سيناريو v17-HR1):", "(دورة 14/06Z — سيناريو v17-HR1 — نافذة 24 ساعة):"),
]

for old, new in reps:
    c = s.count(old)
    assert c == 1, f"توقعت تطابقاً واحداً، وجدت {c}: {old[:60]!r}"
    s = s.replace(old, new)

open(p_dst, "w", encoding="utf-8").write(s)
print(f"✅ {p_dst} تولّد من {p_src} — التحويلات الـ{len(reps)} طُبّقت كلها")
