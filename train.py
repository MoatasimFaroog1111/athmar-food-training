"""
Athmar Vision Count — تدريب نموذج التصنيف على أصناف المطاعم/المقاهي.

يُشغّل على Google Colab (GPU مجاني) أو أي جهاز فيه GPU.

الاستخدام (على Colab):
    !pip install ultralytics
    !python train.py

المتطلبات:
    - مجلد dataset/classes/<صنف>/<صور> (ينتج من download_data.py)
"""
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
CLASSES_DIR = os.path.join(BASE, "dataset", "classes")
EPOCHS = int(os.environ.get("EPOCHS", "50"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
MODEL = os.environ.get("MODEL", "yolov8n-cls.pt")  # nano (أسرع) أو yolov8s-cls.pt (أدق)


def main():
    if not os.path.isdir(CLASSES_DIR):
        print(f"❌ مجلد الأصناف غير موجود: {CLASSES_DIR}")
        print("   شغّل download_data.py أولاً.")
        sys.exit(1)

    classes = sorted(d for d in os.listdir(CLASSES_DIR) if os.path.isdir(os.path.join(CLASSES_DIR, d)))
    if not classes:
        print("❌ لا توجد أصناف. تأكد من بنية dataset/classes/<صنف>/<صور>.")
        sys.exit(1)

    print(f"==> عدد الأصناف: {len(classes)}")
    for c in classes:
        n = len(os.listdir(os.path.join(CLASSES_DIR, c)))
        print(f"   - {c}: {n} صورة")

    model = YOLO(MODEL)

    print(f"\n==> بدء التدريب ({MODEL}, {EPOCHS} epochs, {IMG_SIZE}px) ...")
    results = model.train(
        data=CLASSES_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        project=os.path.join(BASE, "runs"),
        name="athmar-food",
        exist_ok=True,
        device=0,  # GPU
        pretrained=True,
        verbose=True,
    )

    print(f"\n==> أفضل وزن: {results.save_dir}")
    print("==> التدريب اكتمل ✅")


if __name__ == "__main__":
    main()
