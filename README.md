# Athmar Vision Count — تدريب أصناف المطاعم والمقاهي

تدريب نموذج رؤية للتعرف على أدوات ومواد المطاعم والمقاهي، ثم تصديره لتطبيق Athmar Vision Count (Unity Sentis).

## الملفات

| الملف | الوظيفة |
|---|---|
| `classes.txt` | قائمة الأصناف (45 صنفاً: أكل + أدوات + بهارات + مشروبات) |
| `download_data.py` | جلب الصور من Kaggle + فلترة للأصناف |
| `train.py` | تدريب نموذج التصنيف (YOLOv8-cls) |
| `requirements.txt` | المكتبات المطلوبة |

## خطوات التشغيل (على Google Colab — GPU مجاني)

### 1) تجهيز Kaggle API key (مرة واحدة)
1. افتح https://www.kaggle.com/settings/account
2. **Create New Token** → ينزّل `kaggle.json`
3. في Colab:
```python
import os
os.makedirs('/root/.kaggle', exist_ok=True)
# ارفع kaggle.json ثم:
!mv kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
```

### 2) تحميل البيانات
```bash
!pip install kagglehub ultralytics
!python download_data.py
```

### 3) التدريب (GPU)
```bash
!python train.py
```

### 4) النتيجة
- أفضل وزن في `runs/athmar-food/weights/best.pt`
- تقييم الدقة يُعرض في نهاية التدريب (accuracy/confusion matrix)

## ملاحظة مهمة — من التصنيف إلى الكشف

- **الآن**: نموذج تصنيف (يحدد صنف الصورة كاملة).
- **لاحقاً**: نموذج كشف (YOLO-detect) يحدد مواقع كائنات متعددة في نفس الإطار — وهذا ما يحتاجه تطبيق Athmar للعدّ على الرفوف.

المرحلة الحالية تبني أساس التصنيف الصحيح للأصناف، ثم ننتقل للكشف (detection) ببيانات مشروحة.
