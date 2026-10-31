"""
Athmar — تجهيز بيانات الكشف (Detection) من COCO لأدوات المطاعم والمقاهي.

COCO يحتوي bounding boxes حقيقية للأدوات والأكل — مثالي لتدريب نموذج عدّ على الرفوف.

الاستخدام (على Colab GPU):
    !python prepare_detection_coco.py

الناتج: dataset_detect/ (صور + تسميات بصيغة YOLO detection)
"""
import os
import json
import shutil
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"
OUT = os.path.join(BASE, "dataset_detect")
os.makedirs(OUT, exist_ok=True)

# أصناف COCO المستهدفة (المطاعم/المقاهي) — (coco_id, اسم الكلاس)
TARGET = [
    (39, "bottle"),      # قارورة ماء
    (40, "wine_glass"),  # كأس
    (41, "cup"),         # فنجان/كوب
    (42, "fork"),        # شوكة
    (43, "knife"),       # سكين
    (44, "spoon"),       # ملعقة
    (45, "bowl"),        # وعاء/زبدية
    (46, "banana"),      # موز
    (47, "apple"),       # تفاح
    (48, "sandwich"),    # ساندويتش
    (49, "orange"),      # برتقال
    (50, "broccoli"),    # بروكلي
    (51, "carrot"),      # جزر
    (52, "hot_dog"),     # هوت دوغ
    (53, "pizza"),       # بيتزا
    (54, "donut"),       # دونات
    (55, "cake"),        # كيك
    (60, "dining_table"),# طاولة
    (68, "microwave"),   # ميكروويف
    (69, "oven"),        # فرن
    (70, "toaster"),     # محمصة
    (71, "sink"),        # مغسلة
    (72, "refrigerator"),# ثلاجة
]

MAX_IMAGES_PER_CLASS = 300  # حد لتقليل الحجم

def main():
    # 1) تحميل annotations
    ann_zip = os.path.join(BASE, "coco_ann.zip")
    if not os.path.exists(ann_zip):
        print("==> تحميل COCO annotations (~250MB) ...")
        urllib.request.urlretrieve(ANN_URL, ann_zip)
        print("   ✓ تم التحميل")

    ann_dir = os.path.join(BASE, "coco_ann")
    if not os.path.exists(os.path.join(ann_dir, "annotations", "instances_train2017.json")):
        print("==> فك الضغط ...")
        with zipfile.ZipFile(ann_zip) as z:
            z.extract("annotations/instances_train2017.json", ann_dir)

    ann_file = os.path.join(ann_dir, "annotations", "instances_train2017.json")
    with open(ann_file) as f:
        coco = json.load(f)

    # 2) بناء الخرائط
    target_ids = {cid for cid, _ in TARGET}
    coco_to_local = {cid: i for i, (cid, _) in enumerate(TARGET)}
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_img = {img["id"]: img for img in coco["images"]}

    # 3) تجميع annotations حسب الصورة (فقط الأصناف المستهدفة)
    img_anns = {}  # image_id -> [annotations]
    for ann in coco["annotations"]:
        if ann["category_id"] in target_ids and not ann.get("iscrowd", 0):
            img_anns.setdefault(ann["image_id"], []).append(ann)

    print(f"\n==> صور تحتوي أصناف مستهدفة: {len(img_anns)}")

    # 4) تنزيل الصور + كتابة تسميات YOLO
    images_dir = os.path.join(OUT, "images")
    labels_dir = os.path.join(OUT, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    per_class_count = {}
    downloaded = 0
    for img_id, anns in img_anns.items():
        # تحقق من حد كل صنف
        cats_in_img = {a["category_id"] for a in anns}
        skip = all(per_class_count.get(c, 0) >= MAX_IMAGES_PER_CLASS for c in cats_in_img)
        if skip:
            continue

        img = id_to_img[img_id]
        # تنزيل الصورة
        dst_img = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(dst_img):
            try:
                urllib.request.urlretrieve(IMG_URL.format(img_id), dst_img)
            except Exception:
                continue

        w, h = img["width"], img["height"]
        # كتابة التسمية
        lines = []
        for ann in anns:
            cid = ann["category_id"]
            if per_class_count.get(cid, 0) >= MAX_IMAGES_PER_CLASS:
                continue
            local = coco_to_local[cid]
            x, y, bw, bh = ann["bbox"]
            # YOLO format: cx, cy, w, h (normalized)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{local} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            per_class_count[cid] = per_class_count.get(cid, 0) + 1

        if lines:
            with open(os.path.join(labels_dir, f"{img_id}.txt"), "w") as f:
                f.write("\n".join(lines))
            downloaded += 1

        if downloaded % 50 == 0:
            print(f"   ✓ {downloaded} صورة ...")

    print(f"\n==> إجمالي الصور: {downloaded}")
    print("==> توزيع الأصناف:")
    for cid, name in TARGET:
        print(f"   {name}: {per_class_count.get(cid, 0)}")

    # 5) كتابة data.yaml (لتدريب YOLO detection)
    names = [name for _, name in TARGET]
    yaml_content = f"""path: {OUT}
train: images
val: images
names:
""" + "\n".join(f"  {i}: {n}" for i, n in enumerate(names))

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"\n==> data.yaml جاهز في {OUT}/data.yaml")


if __name__ == "__main__":
    main()
