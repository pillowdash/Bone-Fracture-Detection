from pathlib import Path
from PIL import Image
import shutil

ROOTS = [
    Path("data/train"),
    Path("data/val"),
    Path("data/test"),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
QUARANTINE = Path("data/bad_images")


def main():
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    moved = []

    for root in ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        img.load()
                except Exception as e:
                    rel = path.relative_to(Path("data"))
                    dst = QUARANTINE / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dst))
                    moved.append((str(path), str(dst), str(e)))

    print(f"Moved bad files: {len(moved)}")
    for src, dst, err in moved:
        print(f"{src} -> {dst} | {err}")


if __name__ == "__main__":
    main()
