---
language: en
license: mit
tags:
- text-classification
- bert
- resume-analysis
- job-classification
- nlp
datasets:
- resume-dataset
metrics:
- accuracy
- matthews_correlation
model-index:
- name: resume-analyser-bert
  results:
  - task:
      type: text-classification
      name: Resume Classification
    dataset:
      name: Resume Dataset
      type: resume-dataset
    metrics:
    - type: accuracy
      value: 1.0
      name: Validation Accuracy
    - type: matthews_correlation
      value: 1.0
      name: Matthews Correlation Coefficient
---

# Resume Analyser - BERT for Job Category Classification

## Model Description

This is a fine-tuned BERT-base-uncased model for classifying resumes into 25 different job categories. The model achieved **100% validation accuracy** on the test dataset.

## Model Details

- **Model Type:** BERT for Sequence Classification
- **Base Model:** bert-base-uncased
- **Parameters:** 109,501,465 (all trainable)
- **Language:** English
- **License:** MIT
- **Training Data:** 962 resumes from Kaggle Resume Dataset
- **Categories:** 25 job categories

## Intended Use

This model is designed to automatically classify resumes into job categories based on their content. It can be used for:

- Automated resume screening systems
- Job recommendation systems
- HR automation tools
- Resume parsing applications
- Career guidance systems

## Training Data

The model was trained on the [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) containing:
- **Total Samples:** 962 resumes
- **Train/Test Split:** 80/20 (769 training, 193 validation)
- **Categories:** 25 job categories

### Job Categories

Data Science, Java Developer, Testing, DevOps Engineer, Python Developer, Web Developer, HR, Hadoop, Blockchain, ETL Developer, Operations Manager, Sales, Mechanical Engineer, Arts, Database, Electrical Engineering, Health and Fitness, PMO, Business Analyst, DotNet Developer, Automation Testing, Network Security Engineer, SAP Developer, Civil Engineer, Advocate

## Training Procedure

### Training Hyperparameters

- **Base Model:** bert-base-uncased
- **Batch Size:** 4
- **Epochs:** 5
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW (foreach=False)
- **Max Sequence Length:** 200 tokens
- **Warmup Steps:** Linear scheduler
- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)

### Training Results

| Epoch | Training Loss | Validation Loss | Validation Accuracy | MCC Score |
|-------|---------------|-----------------|---------------------|-----------|
| 1     | 2.6037        | 1.1563          | 53.37%             | 0.4993    |
| 2     | 0.9651        | 0.2858          | 98.96%             | 0.9891    |
| 3     | 0.5804        | 0.2782          | 100.00%            | 1.0000    |
| 4     | 0.4473        | 0.2774          | 100.00%            | 1.0000    |
| 5     | 0.3604        | 0.2767          | 100.00%            | 1.0000    |

**Training Time:** ~72 minutes for 5 epochs

## Performance Metrics

- **Validation Accuracy:** 100%
- **Matthews Correlation Coefficient:** 1.0000 (perfect correlation)
- **Final Training Loss:** 0.3604
- **Final Validation Loss:** 0.2767

The model achieved perfect classification on the validation set, correctly identifying all 193 test resumes.

## Usage

### Installation

```bash
pip install transformers torch
```

### Quick Start

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('SwaKyxd/resume-analyser-bert')
tokenizer = BertTokenizer.from_pretrained('SwaKyxd/resume-analyser-bert')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Example resume text
resume_text = """
Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow, NLP
Experience: 3 years in Data Science and AI model development
Projects: Built recommendation systems, sentiment analysis models
Education: Masters in Computer Science
"""

# Tokenize
inputs = tokenizer(
    resume_text,
    return_tensors='pt',
    max_length=200,
    padding='max_length',
    truncation=True
)

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence = probabilities[0][predictions[0]].item()

# Category mapping (example - adjust based on your label encoder)
categories = [
    "Data Science", "Java Developer", "Testing", "DevOps Engineer",
    "Python Developer", "Web Developer", "HR", "Hadoop", "Blockchain",
    "ETL Developer", "Operations Manager", "Sales", "Mechanical Engineer",
    "Arts", "Database", "Electrical Engineering", "Health and Fitness",
    "PMO", "Business Analyst", "DotNet Developer", "Automation Testing",
    "Network Security Engineer", "SAP Developer", "Civil Engineer", "Advocate"
]

print(f"Predicted Category: {categories[predictions.item()]}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Processing

```python
resumes = [
    "Python developer with 5 years experience in Django and Flask...",
    "Experienced data scientist with expertise in machine learning...",
    "Java backend developer skilled in Spring Boot and microservices..."
]

# Tokenize batch
inputs = tokenizer(
    resumes,
    return_tensors='pt',
    max_length=200,
    padding='max_length',
    truncation=True
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

for i, pred in enumerate(predictions):
    print(f"Resume {i+1}: {categories[pred.item()]}")
```

## Limitations and Bias

- **Language:** Model is trained only on English resumes
- **Dataset Size:** Trained on 962 resumes, may not generalize to all resume formats
- **Domain Specific:** Performance may vary on resumes outside the 25 predefined categories
- **Text Format:** Best performance on plain text resumes; may need preprocessing for PDFs/DOCs
- **Perfect Accuracy:** The 100% accuracy suggests possible overfitting; recommend testing on new data

## Ethical Considerations

- This model should be used as an assistive tool, not as the sole decision-maker in hiring processes
- Human oversight is recommended for all automated resume screening
- Be aware of potential biases in the training data that may affect predictions
- Ensure compliance with employment laws and anti-discrimination regulations
- Protect candidate privacy and handle resume data securely

## Model Architecture

```
BertForSequenceClassification(
  (bert): BertModel(
    12 transformer layers
    768 hidden dimensions
    12 attention heads
    110M parameters
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(768 -> 25)
)
```

## Technical Specifications

- **Framework:** PyTorch 2.6.0
- **Transformers:** 4.47.1
- **Tokenizer:** BertTokenizer (bert-base-uncased)
- **Max Sequence Length:** 200 tokens
- **Model Size:** ~436 MB
- **Precision:** FP32

## Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{resume-analyser-bert,
  author = {Sayan Mahalik},
  title = {Resume Analyser - BERT for Job Category Classification},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/SwaKyxd/resume-analyser-bert}
}
```

## Related Resources

- **GitHub Repository:** [Resume-Analyser](https://github.com/Swakyxd/Resume-Analyser)
- **Training Notebook:** Available in the GitHub repository
- **Base Model:** [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- **Dataset:** [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

## Contact

For questions, issues, or feedback:
- GitHub: [Swakyxd/Resume-Analyser](https://github.com/Swakyxd/Resume-Analyser)
- Open an issue on GitHub for bug reports or feature requests

## License

This model is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- BERT paper: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- Kaggle Resume Dataset contributors
- PyTorch team

---

**Model Card Version:** 1.0  
**Last Updated:** November 2025
