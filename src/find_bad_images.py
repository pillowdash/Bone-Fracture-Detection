from pathlib import Path
from PIL import Image

ROOTS = [
    Path("data/train"),
    Path("data/val"),
    Path("data/test"),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    bad_files = []

    for root in ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                try:
                    with Image.open(path) as img:
                        img.verify()  # quick integrity check

                    # reopen and fully load pixel data
                    with Image.open(path) as img:
                        img.load()

                except Exception as e:
                    bad_files.append((str(path), str(e)))

    print(f"Bad files found: {len(bad_files)}")
    for path, err in bad_files:
        print(f"{path} -> {err}")


if __name__ == "__main__":
    main()
