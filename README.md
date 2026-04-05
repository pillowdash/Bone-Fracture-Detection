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

# Download Dataset [get API Key(save to ~/.kaggle/kaggle.json) before Download]
mkdir -p data/raw
kaggle datasets download -d mahmudulhasantasin/fracatlas-original-dataset -p data/raw

# Have to downgrade PyTorch to older version to make sure PyTorch can actually use the GPU:PyTorch 2.5.1 + cu121 ( *NVIDIA Driver Version: 535* )
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121


## Initial Baseline Results

After cleaning corrupted images and training a weighted ResNet18 transfer learning model for 5 epochs, the model achieved the following performance on the test set:

- Accuracy: 68.37%
- Precision: 29.21%
- Recall: 54.63%
- F1-score: 38.06%

These results show that the baseline model can detect a meaningful portion of fracture cases, but precision remains limited, indicating a relatively high false-positive rate. Future improvements will focus on fine-tuning deeper layers, training for more epochs, threshold tuning, and Grad-CAM explainability.
