"""
Athmar Vision Count — جلب بيانات التدريب من Kaggle + فلترة للأصناف المطلوبة.

الاستخدام:
    python download_data.py

المتطلبات:
    pip install kagglehub

ملاحظة: بعض datasets من Kaggle تتطلب API key. لتوليده:
    1. افتح https://www.kaggle.com/settings/account
    2. أنشئ API Token → ينزّل kaggle.json
    3. ضعه في ~/.kaggle/kaggle.json
"""
import os
import shutil
import subprocess
import sys

# الأصناف التي نريدها (من classes.txt) — نطابق أسماء فرعية مع Food-101
# Food-101 يغطي الأكل فقط؛ أدوات المطبخ (كوب، ملعقة...) تأتي من COCO لاحقاً.
FOOD_CLASSES = {
    "fried_chicken": ["fried chicken", "fried_rice", "fried", "chicken"],
    "chicken_wings": ["chicken_curry", "chicken_wings", "chicken_quesadilla"],
    "kebab": ["kebab", "sashimi", "steak", "grilled_salmon"],
    "beef_steak": ["steak", "filet_mignon", "beef"],
    "burger_patty": ["hamburger", "cheeseburger"],
    "fish": ["grilled_salmon", "fish_and_chips", "sashimi"],
    "bread": ["bread", "french_toast", "waffles", "pancakes"],
    "burger_bun": ["hamburger", "cheeseburger"],
    "croissant": ["croissant", "bread"],
    "onion": ["onion_rings", "onion"],
    "tomato": ["bruschetta", "caprese", "tomato"],
    "potato": ["french_fries", "hash_browns", "mashed_potatoes", "potato"],
    "lemon": ["lemon", "ceviche"],
    "garlic": ["garlic", "garlic_bread"],
    "juice_glass": ["orange_juice", "apple_juice", "lemonade", "smoothie"],
    "soda_can": ["soda", "cola", "lemonade"],
    "tea": ["tea", "hot_tea"],
    "espresso": ["espresso", "coffee", "latte", "cappuccino"],
    "arabic_coffee": ["coffee", "espresso"],
    "turmeric": ["curry", "butter_chicken"],
    "curry_powder": ["curry", "butter_chicken", "chicken_curry"],
}

DATASET = "kmader/food41"  # Food-41: نسخة أصغر وأسرع من Food-101

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("==> تحميل dataset من Kaggle ...")
    try:
        import kagglehub  # noqa
    except ImportError:
        print("تثبيت kagglehub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub  # noqa

    path = kagglehub.dataset_download(DATASET)
    print(f"==> تم التحميل إلى: {path}")

    # فحص بنية dataset
    print("==> محتوى dataset:")
    for root, dirs, files in os.walk(path):
        depth = root[len(path):].count(os.sep)
        if depth < 2:
            print("   " + os.path.relpath(root, path) + f" ({len(files)} files)")

    # نسخ الأصناف المطلوبة إلى مجلد منظم (classes/اسم/صور)
    print("\n==> فلترة الأصناف المطلوبة ...")
    classes_dir = os.path.join(OUT_DIR, "classes")
    os.makedirs(classes_dir, exist_ok=True)

    copied = 0
    for target, candidates in FOOD_CLASSES.items():
        target_dir = os.path.join(classes_dir, target)
        os.makedirs(target_dir, exist_ok=True)
        for candidate in candidates:
            src = os.path.join(path, "images", candidate) if os.path.exists(os.path.join(path, "images", candidate)) else os.path.join(path, candidate)
            if os.path.isdir(src):
                for f in os.listdir(src)[:200]:  # حد 200 صورة/صنف
                    shutil.copy(os.path.join(src, f), os.path.join(target_dir, f))
                    copied += 1
                print(f"   ✓ {target} <- {candidate}")
                break

    print(f"\n==> إجمالي الصور المنسوخة: {copied}")
    print(f"==> المجلد النهائي: {classes_dir}")


if __name__ == "__main__":
    main()
