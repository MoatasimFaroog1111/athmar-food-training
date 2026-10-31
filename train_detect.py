"""
Athmar — تدريب نموذج الكشف (Detection) على أدوات المطاعم/المقاهي.

يُشغّل على Colab GPU بعد prepare_detection_coco.py.

الاستخدام:
    !python train_detect.py
"""
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "dataset_detect", "data.yaml")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
MODEL = os.environ.get("MODEL", "yolov8n.pt")  # nano detect


def main():
    if not os.path.exists(DATA_YAML):
        print(f"❌ data.yaml غير موجود: {DATA_YAML}")
        print("   شغّل prepare_detection_coco.py أولاً.")
        sys.exit(1)

    print(f"==> بدء تدريب الكشف ({MODEL}, {EPOCHS} epochs, {IMG_SIZE}px) ...")
    model = YOLO(MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        project=os.path.join(BASE, "runs"),
        name="athmar-detect",
        exist_ok=True,
        device=0,
        pretrained=True,
    )

    print(f"\n==> أفضل وزن: {results.save_dir}/weights/best.pt")
    print("==> التدريب اكتمل ✅")


if __name__ == "__main__":
    main()
