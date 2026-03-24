"""
Quick Test Script - Resume Classifier
Simple script to quickly test the model with predefined samples.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

# Job categories
CATEGORIES = [
    "Data Science", "Java Developer", "Testing", "DevOps Engineer",
    "Python Developer", "Web Developer", "HR", "Hadoop", "Blockchain",
    "ETL Developer", "Operations Manager", "Sales", "Mechanical Engineer",
    "Arts", "Database", "Electrical Engineering", "Health and Fitness",
    "PMO", "Business Analyst", "DotNet Developer", "Automation Testing",
    "Network Security Engineer", "SAP Developer", "Civil Engineer", "Advocate"
]

def load_model():
    """Load the trained model."""
    print("Loading model...")
    
    model_path = 'models/saved_models/resume_analyser_bert'
    if not os.path.exists(model_path):
        model_path = 'SwaKyxd/resume-analyser-bert'
        print("Loading from Hugging Face Hub...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict(model, tokenizer, device, text):
    """Make prediction for resume text."""
    inputs = tokenizer(text, return_tensors='pt', max_length=200, 
                      padding='max_length', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    category_id = pred.item()
    confidence = probs[0][category_id].item()
    
    # Get top 3
    top3_idx = torch.topk(probs[0], 3).indices
    top3 = [(CATEGORIES[i], probs[0][i].item()) for i in top3_idx]
    
    return CATEGORIES[category_id], confidence, top3

def main():
    """Run quick tests."""
    model, tokenizer, device = load_model()
    
    # Test samples
    tests = [
        ("Data Science Resume", """
            Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Data Analysis,
            Statistics, SQL, Pandas, NumPy, 4 years experience in AI and ML projects
        """),
        
        ("Python Developer Resume", """
            Python, Django, Flask, REST APIs, PostgreSQL, Docker, AWS, 
            5 years backend development, microservices architecture
        """),
        
        ("Java Developer Resume", """
            Java, Spring Boot, Hibernate, Microservices, MySQL, Maven,
            6 years enterprise application development
        """),
        
        ("Web Developer Resume", """
            JavaScript, React, HTML, CSS, Node.js, MongoDB, Redux,
            4 years full-stack web development
        """),
        
        ("Mechanical Engineer Resume", """
            AutoCAD, SolidWorks, CAD design, Manufacturing, Thermodynamics,
            5 years mechanical design and analysis
        """)
    ]
    
    print("="*70)
    print(" TESTING RESUME CLASSIFIER")
    print("="*70)
    
    for i, (name, resume) in enumerate(tests, 1):
        print(f"\n{i}. Testing: {name}")
        print("-" * 70)
        
        category, confidence, top3 = predict(model, tokenizer, device, resume)
        
        print(f"   Predicted: {category}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"\n   Top 3 Predictions:")
        for rank, (cat, score) in enumerate(top3, 1):
            print(f"      {rank}. {cat:25s} {score:.1%}")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
