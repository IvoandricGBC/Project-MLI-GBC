# Distance Estimation – Modular Pipeline (V3)

## 1. Project Overview
This project implements a **modular deep learning pipeline** for distance estimation using image data.  
The workflow is designed to support both **full-size images** and **cropped images** (with configurable padding), enabling comparative training and evaluation.  

The architecture is divided into independent modules for configuration, data loading, model definition, metrics, training, evaluation, and prediction.  
This modularity allows for easy updates, debugging, and scalability in an academic or industrial setting.

---

## 2. Features
- **Google Colab integration** with Google Drive mounting.
- **Two input modes**: full images and cropped images with ±30 px padding.
- **Configurable hyperparameters** from a single file (`config.py`).
- **Automatic training safeguards**:
  - Minimum percentage of new IDs required to trigger retraining.
  - Validation MAE threshold to discard weak models.
- **Support for freezing and unfreezing layers** during training.
- **Evaluation with multiple α values** to test model stability.
- **Exportable CSV reports** for true vs. predicted distances.
- **Built-in charts** for visual analysis of model performance.

---

## 3. Folder Structure
The project expects the following structure in the root folder (generated from the V7 prep-data pipeline):

```
project_root/
│
├── images_full/                 # Full-size PNG images
├── cropped_images_pad30/        # Cropped images (±30 px padding)
│
├── data/csv/
│   ├── full_train.csv            # Training dataset (full images)
│   ├── cropped_train.csv         # Training dataset (cropped images)
│   ├── holdout200.csv            # Hold-out dataset for evaluation
│   └── class_mapping.json        # JSON mapping between class labels and IDs
│
├── ckpts/                        # Model checkpoints (auto-created if missing)
├── state/                        # Training state and logs (auto-created if missing)
│
├── config.py                     # Global constants, paths, and hyperparameters
├── data.py                       # Data loading and augmentation logic
├── models.py                     # Model architectures
├── utils/
│   ├── metrics.py                # Custom evaluation metrics
│   └── guards.py                 # Training safeguards
├── train.py                      # Training loop
├── eval.py                       # Model evaluation
└── predict.py                    # Inference pipeline
```

---

## 4. Requirements
Install the required dependencies in Colab or your local environment:

```bash
pip install torch torchvision torchaudio
pip install timm albumentations pytorch-lightning
pip install scikit-learn matplotlib opencv-python
```

---

## 5. Usage

### Step 1 – Mount Google Drive
In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Update `ROOT` in `config.py` to match your project directory.

---

### Step 2 – Training
Run the training pipeline:
```bash
python train.py
```
Options:
- `--force` → bypass safeguards and retrain even without new data.

---

### Step 3 – Evaluation
Evaluate the trained model with multiple α values:
```bash
python eval.py
```

---

### Step 4 – Prediction
Generate predictions for a list of image IDs:
```bash
python predict.py --ids 101 205 333
```

---

### Step 5 – Reports & Charts
- **True vs. Predicted CSV**: Exported after evaluation.
- **Error Histogram**: Visualizes absolute error distribution.
- **Scatter Plot**: Shows predicted vs. actual distances.

---

## 6. Example Workflow (One-Click Run)
1. **TRAIN** – Force run, logs safeguards.
2. **EVALUATE** – α sweep.
3. **EXPORT CSV** – True vs. Pred.
4. **CHARTS** – Generate histogram & scatter plot.

---

## 7. Expected Outputs
- **Model Checkpoints** in `ckpts/`.
- **Evaluation CSV** in `data/csv/`.
- **Matplotlib Figures** in output folder.
- Training logs with metrics (MAE, RMSE, R²).

---

## 8. Author & Acknowledgments
This modular pipeline was designed for **academic and industrial applications** in computer vision distance estimation.  
It can be easily extended to other regression-based image analysis tasks.
