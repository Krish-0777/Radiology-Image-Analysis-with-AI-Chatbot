# 🩺 RadVision AI — Radiology Image Analysis Platform

A deep learning platform for classifying radiology scans into:
**Normal · Cancer · Fracture · COVID-19**

Built with ResNet-50 CNN + Streamlit UI + Gemini AI chatbot.

---

## 📁 Project Structure

```
radvision/
├── app.py              # Streamlit UI (main entry point)
├── trainer.py          # Model training script
├── prepare_data.py     # Kaggle data downloader & organizer
├── requirements.txt    # Python dependencies
├── outputs/
│   └── best_model.pt   # Saved after training
└── data/
    ├── train/
    │   ├── Normal/
    │   ├── Cancer/
    │   ├── Fracture/
    │   └── COVID-19/
    ├── val/
    └── test/
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.32.0
google-genai>=1.0.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
kaggle>=1.6.0
```

---

### 2. Set Up Kaggle API

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. This downloads `kaggle.json`
3. Place it in the correct location:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<YourUser>\.kaggle\kaggle.json`
4. On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

---

### 3. Download & Prepare Data

```bash
python prepare_data.py --output_dir ./data --images_per_class 2000
```

This downloads from Kaggle and organizes into `train/val/test` splits:

| Dataset | Kaggle Slug | Used For |
|---|---|---|
| COVID-19 Radiography DB | `tawsifurrahman/covid19-radiography-database` | COVID-19 + Normal |
| Chest X-Ray (Pneumonia) | `paultimothymooney/chest-xray-pneumonia` | Cancer proxy + Normal |
| MURA Musculoskeletal | `devendra1bhatt/mura-musculoskeletal-radiographs` | Fracture |

> **Note on Cancer class:** For production, replace with a proper cancer dataset such as:
> - `nih-chest-xrays/data` (NIH Chest X-Ray 14 — 112,000 images)
> - `andrewmvd/lung-and-colon-cancer-histopathological-images`

---

### 4. Train the Model

```bash
python trainer.py --data_dir ./data --epochs 30 --batch_size 32 --lr 1e-4
```

**Training features:**
- ResNet-50 with ImageNet pre-training (transfer learning)
- Frozen backbone → fine-tune last 2 blocks + custom head
- Mixed precision (AMP) for faster GPU training
- Weighted random sampler for class imbalance
- Label smoothing + Dropout regularization
- Early stopping + ReduceLROnPlateau scheduler
- Best model auto-saved to `outputs/best_model.pt`

**Expected training time:**
- GPU (RTX 3080): ~25 min for 30 epochs, 8000 images/class
- CPU: ~4-6 hours (not recommended)

---

### 5. Load Trained Weights in app.py

After training, open `app.py` and uncomment this section in `load_model()`:

```python
weights_path = "outputs/best_model.pt"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
```

---

### 6. Run the App

```bash
streamlit run app.py
```

Then:
1. Open [http://localhost:8501](http://localhost:8501)
2. Enter your **Gemini API Key** in the sidebar
   - Get one free at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
3. Upload a chest X-ray image (PNG/JPG)
4. Click **Run AI Detection**
5. Ask the Gemini chatbot about the findings

---

## 🔬 Model Architecture

```
Input Image (224×224×3)
        ↓
ResNet-50 Backbone (ImageNet pretrained)
   Layer1 → Layer2 → [Layer3 frozen] → Layer4 (fine-tuned)
        ↓
   Global Average Pool → 2048-d feature vector
        ↓
   Dropout(0.4) → Linear(2048→256) → ReLU → Dropout(0.3)
        ↓
   Linear(256→4) → Softmax
        ↓
[Normal | Cancer | Fracture | COVID-19]
```

---

## 🎯 Expected Performance

With 2000 images/class and 30 epochs:

| Class | Expected Accuracy |
|---|---|
| Normal | ~90-95% |
| COVID-19 | ~88-94% |
| Cancer | ~82-88% |
| Fracture | ~80-87% |
| **Overall** | **~85-92%** |

Performance improves significantly with more data (5000+ images/class).

---

## ⚠️ Disclaimer

**This tool is for research and educational purposes only.**
It is NOT a medical device. Always consult a qualified radiologist
for clinical diagnosis.

---

## 📜 License

MIT License. Kaggle dataset licenses apply to respective datasets.
