"""
فلترة Food-101 إلى الأصناف المطلوبة (من classes.txt).
Food-101 يغطي الأكل فقط؛ الأدوات والمشروبات والبهارات تحتاج مصادر أخرى.
"""
import os
import shutil
import sys

SRC = os.path.expanduser("~/.cache/kagglehub/datasets/kmader/food41/versions/5/images")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "classes")
os.makedirs(DST, exist_ok=True)

# مطابقة أصنافنا (الهدف) -> أسماء مجلدات Food-101 (المصدر)
MAPPING = {
    "beef_steak": ["steak", "filet_mignon", "prime_rib"],
    "chicken_wings": ["chicken_wings"],
    "fried_chicken": ["fried_calamari", "fried_rice", "chicken_quesadilla"],
    "burger_patty": ["hamburger", "club_sandwich", "pulled_pork_sandwich"],
    "fish": ["grilled_salmon", "fish_and_chips", "sashimi", "ceviche"],
    "kebab": ["sashimi", "steak", "pork_chop"],
    "bread": ["french_toast", "pancakes", "waffles", "bread_pudding"],
    "burger_bun": ["hamburger", "hot_dog", "club_sandwich"],
    "onion": ["onion_rings"],
    "tomato": ["caprese_salad", "bruschetta", "caesar_salad"],
    "potato": ["french_fries", "gnocchi"],
    "garlic": ["garlic_bread"],
    "curry_powder": ["chicken_curry", "pad_thai"],
    "turmeric": ["chicken_curry", "samosa"],
    "croissant": ["croque_madame", "beignets", "donuts"],
}

PER_CLASS = 300

def main():
    total = 0
    for target, sources in MAPPING.items():
        target_dir = os.path.join(DST, target)
        os.makedirs(target_dir, exist_ok=True)
        copied = 0
        for src in sources:
            src_dir = os.path.join(SRC, src)
            if not os.path.isdir(src_dir):
                continue
            files = sorted(os.listdir(src_dir))[:PER_CLASS]
            for f in files:
                dst = os.path.join(target_dir, f"{src}_{f}")
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(src_dir, f), dst)
                    copied += 1
        total += copied
        print(f"  ✓ {target}: {copied} صورة")

    print(f"\n==> إجمالي الصور المنسوخة: {total}")
    print(f"==> المجلد: {DST}")

    # عرض الأصناف المغطاة وغير المغطاة
    print("\n==> أصناف مطلوبة لم تُغطَّ من Food-101 (تحتاج مصادر أخرى):")
    covered = set(MAPPING.keys())
    all_classes = set()
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.txt"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "|" in line:
            all_classes.add(line.split("|")[0].strip())
    for c in sorted(all_classes - covered):
        print(f"  - {c}")


if __name__ == "__main__":
    main()
