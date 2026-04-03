from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_ROOT = Path("data/raw/FracAtlas")
CSV_PATH = RAW_ROOT / "dataset.csv"
IMAGES_ROOT = RAW_ROOT / "images"
OUTPUT_ROOT = Path("data")

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def find_image_path(image_name: str) -> Path | None:
    """
    Search recursively for an image file under IMAGES_ROOT by filename.
    """
    matches = list(IMAGES_ROOT.rglob(image_name))
    if matches:
        return matches[0]
    return None


def normalize_label(value):
    """
    Convert possible label values into 'fractured' or 'normal'.
    Adjust if needed after inspecting dataset.csv.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"fractured", "fracture", "1", "true", "yes"}:
            return "fractured"
        if v in {"non_fractured", "non-fractured", "normal", "0", "false", "no"}:
            return "normal"

    if value == 1:
        return "fractured"
    if value == 0:
        return "normal"

    return None


def make_output_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["fractured", "normal"]:
            (OUTPUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print("CSV loaded.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Try to identify image filename column
    candidate_image_cols = ["image_id", "image", "image_name", "filename", "file_name", "path"]
    image_col = None
    for col in candidate_image_cols:
        if col in df.columns:
            image_col = col
            break

    if image_col is None:
        raise ValueError(
            f"Could not find image filename column. Available columns: {df.columns.tolist()}"
        )

    # Label column
    if "fractured" not in df.columns:
        raise ValueError(
            f"Could not find 'fractured' column. Available columns: {df.columns.tolist()}"
        )

    # Keep only rows with usable labels
    df["class_name"] = df["fractured"].apply(normalize_label)
    df = df[df["class_name"].notna()].copy()

    print("\nClass counts after normalization:")
    print(df["class_name"].value_counts())

    # Resolve image paths
    resolved_paths = []
    missing = []

    for _, row in df.iterrows():
        image_value = str(row[image_col]).strip()
        img_path = find_image_path(image_value)
        if img_path is None:
            missing.append(image_value)
            resolved_paths.append(None)
        else:
            resolved_paths.append(img_path)

    df["resolved_path"] = resolved_paths
    missing_count = df["resolved_path"].isna().sum()

    print(f"\nResolved images: {len(df) - missing_count}")
    print(f"Missing images: {missing_count}")

    if missing_count > 0:
        print("\nFirst 20 missing image names:")
        for name in missing[:20]:
            print(" ", name)

    df = df[df["resolved_path"].notna()].copy()

    # Stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=df["class_name"],
        random_state=RANDOM_STATE
    )

    val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative),
        stratify=temp_df["class_name"],
        random_state=RANDOM_STATE
    )

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    make_output_dirs()

    def copy_split(split_df, split_name):
        copied = 0
        for _, row in split_df.iterrows():
            src = Path(row["resolved_path"])
            dst = OUTPUT_ROOT / split_name / row["class_name"] / src.name
            shutil.copy2(src, dst)
            copied += 1
        print(f"Copied {copied} files to {split_name}")

    copy_split(train_df, "train")
    copy_split(val_df, "val")
    copy_split(test_df, "test")

    print("\nDone.")
    print("You should now have:")
    print("data/train/fractured, data/train/normal, data/val/..., data/test/...")

    print("\nFinal split class counts:")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}")
        print(split_df["class_name"].value_counts())


if __name__ == "__main__":
    main()
