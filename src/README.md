# Model Testing Scripts

This directory contains scripts to test and demonstrate the Resume Analyser model.

## Available Scripts

### 1. `test_model.py` - Interactive Testing Tool

**Full-featured interactive script** with visualization and multiple testing modes.

#### Features:
- ✅ Test predefined sample resumes
- ✅ Enter custom resume text
- ✅ Batch test all samples
- ✅ Visualize predictions with charts
- ✅ Show top N predictions with confidence scores
- ✅ Save visualizations to `outputs/results/`

#### Usage:
```bash
python src/test_model.py
```

#### Options:
1. **Test sample resumes** - Choose from 7 predefined resume samples
2. **Enter custom resume text** - Type or paste your own resume
3. **Batch test all samples** - Test all samples at once
4. **Exit** - Close the program

#### Sample Output:
```
======================================================================
 PREDICTION RESULTS
======================================================================

📄 Resume Text (first 200 chars):
   Skills: Python, R, Machine Learning, Deep Learning, TensorFlow...

🎯 Predicted Category: Data Science
   Category ID: 0
   Confidence: 98.50%
   Confidence Level: Very High ✨

📊 Top 5 Predictions:
   1. Data Science                    98.50% ████████████████████████████████████████
   2. Python Developer                 0.80% █
   3. Testing                          0.20% 
   4. ETL Developer                    0.15% 
   5. Database                         0.10%
```

### 2. `quick_test.py` - Quick Verification Script

**Simple script** to quickly verify the model is working correctly.

#### Features:
- ✅ Tests 5 predefined samples
- ✅ Shows top 3 predictions for each
- ✅ Fast execution (~10 seconds)
- ✅ No visualization or user input

#### Usage:
```bash
python src/quick_test.py
```

#### Sample Output:
```
======================================================================
 TESTING RESUME CLASSIFIER
======================================================================

1. Testing: Data Science Resume
----------------------------------------------------------------------
   Predicted: Data Science
   Confidence: 54.4%

   Top 3 Predictions:
      1. Data Science              54.4%
      2. Electrical Engineering    15.1%
      3. ETL Developer             4.5%
```

### 3. `check_gpu.py` - GPU Verification

Verify CUDA/GPU setup is working correctly.

#### Usage:
```bash
python src/check_gpu.py
```

## Sample Resumes Included

The interactive script includes 7 professionally crafted sample resumes:

1. **Data Scientist** - ML, DL, Python, R, TensorFlow, PyTorch
2. **Python Developer** - Django, Flask, FastAPI, REST APIs
3. **DevOps Engineer** - AWS, Docker, Kubernetes, CI/CD
4. **Java Developer** - Spring Boot, Hibernate, Microservices
5. **Web Developer** - React, Node.js, JavaScript, HTML/CSS
6. **Mechanical Engineer** - AutoCAD, SolidWorks, CAD, FEA
7. **HR Professional** - Recruitment, Employee Relations, HRIS

## Requirements

Make sure you have installed:
```bash
pip install torch transformers tokenizers matplotlib seaborn
```

## Tips for Best Results

### Resume Text Guidelines:
- ✅ Include **technical skills** relevant to the job
- ✅ Mention **years of experience**
- ✅ List **projects and achievements**
- ✅ Add **education and certifications**
- ✅ Use **keywords** specific to the field
- ❌ Avoid very short text (minimum 50 words recommended)

### Example Good Resume Text:
```
Skills: Python, Django, REST APIs, PostgreSQL, Docker
Experience: 5 years as Backend Developer
Projects: E-commerce platform, Microservices architecture
Education: Bachelor's in Computer Science
Achievements: Optimized API performance by 60%
```

### Example Poor Resume Text:
```
I know programming
```

## Interpreting Confidence Scores

| Confidence | Interpretation | Action |
|-----------|---------------|---------|
| > 90% | Very High ✨ | Model is very confident |
| 70-90% | High ✓ | Good prediction |
| 50-70% | Moderate ~ | Consider top 3 predictions |
| < 50% | Low ⚠ | Review resume content |

## Troubleshooting

### Model not loading
```bash
# Check if model exists locally
ls models/saved_models/resume_analyser_bert/

# If not, it will download from Hugging Face automatically
```

### CUDA out of memory
The scripts use minimal memory and should work on any GPU with 2GB+ VRAM.

### Import errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

## Output Files

### Visualizations
- Location: `outputs/results/`
- Format: PNG images (300 DPI)
- Naming: `prediction_YYYYMMDD_HHMMSS.png`

## Examples

### Test a Custom Resume
```python
# Run interactive mode
python src/test_model.py

# Choose option 2
# Paste your resume text
# Type 'END' when done
# View results with visualization
```

### Quick Batch Test
```python
# Run quick test
python src/quick_test.py

# Results show in ~10 seconds
# No user input needed
```

## Performance

- **Loading Time:** ~5-10 seconds (first time)
- **Prediction Time:** ~50-100ms per resume (GPU)
- **Prediction Time:** ~200-500ms per resume (CPU)
- **Batch Processing:** Faster for multiple resumes

## License

Same as main project (MIT License)
