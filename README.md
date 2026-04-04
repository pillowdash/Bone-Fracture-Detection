# Bone Fracture Detection

Binary classification project for bone X-ray images:
- Fractured
- Normal

## Project Structure

- `data/`: raw, processed, train, val, test (not added into GitHub, download yourself)
- `notebooks/`: EDA, training, Grad-CAM
- `src/`: Python source files
- `outputs/`: models, plots, predictions


## Setup (Python virtual environment)

### Linux / macOS
```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Download Dataset (get API Key before Download)
mkdir -p data/raw
kaggle datasets download -d mahmudulhasantasin/fracatlas-original-dataset -p data/raw

# Have to downgrade PyTorch to older version to make sure PyTorch can actually use the GPU:PyTorch 2.5.1 + cu121 (for my old laptop)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
