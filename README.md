# Resume Analyser - BERT-based Job Category Classifier

A deep learning model that automatically classifies resumes into 25 job categories using BERT (Bidirectional Encoder Representations from Transformers).

## 🎯 Project Overview

This project implements a state-of-the-art NLP model for resume classification. The model analyzes resume text and predicts the most suitable job category with **100% validation accuracy**.

### Key Features
- ✅ **BERT-base-uncased** fine-tuned for resume classification
- ✅ **25 Job Categories** classification
- ✅ **100% Validation Accuracy** achieved
- ✅ **GPU Accelerated** training on NVIDIA RTX 3060
- ✅ **962 Resume Dataset** from Kaggle
- ✅ **Pre-trained Model Available** on [🤗 Hugging Face Hub](https://huggingface.co/SwaKyxd/resume-analyser-bert)
- ✅ **REST API** for easy integration and deployment

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 100% |
| Matthews Correlation Coefficient | 1.0000 |
| Final Training Loss | 0.3604 |
| Final Validation Loss | 0.2767 |

### Training Progress
- **Epoch 1:** 53.37% accuracy → 98.96% accuracy (Epoch 2)
- **Epochs 3-5:** Maintained 100% accuracy

## 🎓 Job Categories

The model classifies resumes into 25 categories:
Data Science, Java Developer, Testing, DevOps Engineer, Python Developer, Web Developer, HR, Hadoop, Blockchain, ETL Developer, Operations Manager, Sales, Mechanical Engineer, Arts, Database, Electrical Engineering, Health and Fitness, PMO, Business Analyst, DotNet Developer, Automation Testing, Network Security Engineer, SAP Developer, Civil Engineer, Advocate

## 🛠️ Technical Stack

- **Framework:** PyTorch 2.6.0 + CUDA 12.4
- **Transformer Library:** Hugging Face Transformers 4.47.1
- **Model:** BERT-base-uncased (109M parameters)
- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
- **Language:** Python 3.10

## 📁 Project Structure

```
resume-analyser/
├── api/                             # FastAPI REST API
│   ├── app.py                       # API application
│   ├── test_api.py                  # API tests
│   ├── requirements.txt             # API dependencies
│   ├── README.md                    # API documentation
│   └── QUICKSTART.md                # Quick start guide
├── data/
│   └── raw/
│       └── resume_dataset/          # Kaggle resume dataset
├── models/
│   └── saved_models/
│       └── resume_analyser_bert/    # Trained model files
│           ├── pytorch_model.bin    # Model weights (436MB)
│           ├── config.json          # Model configuration
│           ├── tokenizer files      # BERT tokenizer
│           ├── training_metrics.json # Training history
│           └── model_info.txt       # Model details
├── notebooks/
│   └── resume_analyser.ipynb        # Training notebook
├── src/
│   └── check_gpu.py                 # GPU verification script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (6GB+ VRAM recommended)
- Anaconda/Miniconda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Swakyxd/Resume-Analyser.git
cd Resume-Analyser
```

2. **Download the trained model from Hugging Face**
```bash
# Install Hugging Face Hub
pip install huggingface-hub

# Download the model (436MB)
huggingface-cli download SwaKyxd/resume-analyser-bert --local-dir models/saved_models/resume_analyser_bert
```

3. **Create conda environment**
```bash
conda create -n DL-GPU python=3.10
conda activate DL-GPU
```

4. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.1 tokenizers==0.21.0
pip install pandas scikit-learn matplotlib seaborn kagglehub jupyter
```

5. **Verify GPU setup**
```bash
python src/check_gpu.py
```

### 📥 Dataset

The model was trained on the [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) from Kaggle (962 resumes, 25 categories).

Download using:
```python
import kagglehub
path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")
```

## 💻 Usage

### 🚀 Quick Start: Using the REST API

The easiest way to use the model is through the FastAPI REST API:

1. **Start the API:**
```bash
cd api
python app.py
```

2. **Access the interactive docs:** http://localhost:8000/docs

3. **Make predictions:**
```python
import requests

resume = "Python developer with 5 years experience in Django and Flask"
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": resume}
)
print(response.json())
```

📖 **Full API documentation:** See [api/README.md](api/README.md)

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/resume_analyser.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Text cleaning and tokenization
- BERT model fine-tuning
- Training with GPU acceleration
- Model evaluation and saving

### Loading the Trained Model

**Option 1: Load from Hugging Face Hub (Recommended)**
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load directly from Hugging Face
model = BertForSequenceClassification.from_pretrained('SwaKyxd/resume-analyser-bert')
tokenizer = BertTokenizer.from_pretrained('SwaKyxd/resume-analyser-bert')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
```

**Option 2: Load from local files**
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load from downloaded model directory
model = BertForSequenceClassification.from_pretrained('models/saved_models/resume_analyser_bert')
tokenizer = BertTokenizer.from_pretrained('models/saved_models/resume_analyser_bert')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
```

### Making Predictions

```python
# Sample resume text
resume_text = """
Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow
Experience: 3 years in Data Science and AI model development
"""

# Preprocess
inputs = tokenizer(resume_text, 
                   return_tensors='pt', 
                   max_length=200, 
                   padding='max_length', 
                   truncation=True)

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    
print(f"Predicted Category ID: {predictions.item()}")
```

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | bert-base-uncased |
| Batch Size | 4 |
| Epochs | 5 |
| Learning Rate | 2e-5 |
| Max Sequence Length | 200 tokens |
| Optimizer | AdamW (foreach=False) |
| Scheduler | Linear warmup |
| Train/Test Split | 80/20 (769/193 samples) |

## 📈 Training Details

- **Training Time:** ~72 minutes (5 epochs)
- **Total Parameters:** 109,501,465 (all trainable)
- **Training Batches:** 193 per epoch
- **Validation Batches:** 49 per epoch
- **GPU Memory:** Optimized for 6GB VRAM

## 🔬 Model Architecture

```
BertForSequenceClassification
├── BERT Base (12 layers, 768 hidden, 12 attention heads)
├── Dropout (p=0.1)
└── Linear Classifier (768 → 25 classes)
```

## 📊 Results & Metrics

The model achieved exceptional performance:
- Perfect classification on validation set (100% accuracy)
- Matthews Correlation Coefficient: 1.0 (perfect correlation)
- Rapid convergence: 98.96% accuracy by Epoch 2
- Consistent performance across all 25 categories

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the BERT implementation
- [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) for the training data
- BERT paper: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔮 Future Improvements

- [ ] Add support for multi-language resumes
- [ ] Implement resume parsing for structured data extraction
- [ ] Create web API for model deployment
- [ ] Add confidence scores and top-k predictions
- [ ] Expand dataset with more resume samples
- [ ] Deploy as a web application

---

**Note:** This model is for educational and research purposes. Ensure compliance with data privacy regulations when using in production.
