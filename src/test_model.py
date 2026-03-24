"""
Interactive Resume Classification Demo
Test the trained BERT model with various resume samples and visualize predictions.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Job categories
CATEGORIES = [
    "Data Science", "Java Developer", "Testing", "DevOps Engineer",
    "Python Developer", "Web Developer", "HR", "Hadoop", "Blockchain",
    "ETL Developer", "Operations Manager", "Sales", "Mechanical Engineer",
    "Arts", "Database", "Electrical Engineering", "Health and Fitness",
    "PMO", "Business Analyst", "DotNet Developer", "Automation Testing",
    "Network Security Engineer", "SAP Developer", "Civil Engineer", "Advocate"
]

class ResumeClassifier:
    def __init__(self, model_path='models/saved_models/resume_analyser_bert'):
        """Initialize the classifier with trained model."""
        print("="*70)
        print(" Resume Classification Demo")
        print("="*70)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading model...")
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}")
            print("Loading from Hugging Face Hub...")
            model_path = 'SwaKyxd/resume-analyser-bert'
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully!")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ready for predictions\n")
    
    def predict(self, resume_text, show_top_n=5):
        """
        Predict job category for a resume.
        
        Args:
            resume_text: Resume text to classify
            show_top_n: Number of top predictions to show
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        inputs = self.tokenizer(
            resume_text,
            return_tensors='pt',
            max_length=200,
            padding='max_length',
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Get results
        category_id = prediction.item()
        category = CATEGORIES[category_id]
        confidence = probabilities[0][category_id].item()
        
        # Get all scores
        all_scores = {
            CATEGORIES[i]: float(probabilities[0][i].item())
            for i in range(len(CATEGORIES))
        }
        
        # Get top N predictions
        top_indices = torch.topk(probabilities[0], show_top_n).indices
        top_predictions = [
            (CATEGORIES[i.item()], probabilities[0][i].item())
            for i in top_indices
        ]
        
        return {
            'category': category,
            'category_id': category_id,
            'confidence': confidence,
            'all_scores': all_scores,
            'top_predictions': top_predictions
        }
    
    def visualize_prediction(self, result, resume_text):
        """Visualize prediction results with bar charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Top 10 predictions
        top_10 = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:10]
        categories = [item[0] for item in top_10]
        scores = [item[1] * 100 for item in top_10]
        
        colors = ['#2ecc71' if cat == result['category'] else '#3498db' for cat in categories]
        
        ax1.barh(range(len(categories)), scores, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(categories)))
        ax1.set_yticklabels(categories)
        ax1.set_xlabel('Confidence (%)', fontsize=12)
        ax1.set_title('Top 10 Predicted Categories', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add percentage labels
        for i, score in enumerate(scores):
            ax1.text(score + 1, i, f'{score:.1f}%', va='center', fontsize=10)
        
        # Plot 2: Confidence distribution
        all_scores_sorted = sorted(result['all_scores'].values(), reverse=True)
        ax2.plot(range(1, len(all_scores_sorted) + 1), 
                [s * 100 for s in all_scores_sorted], 
                marker='o', linewidth=2, markersize=6, color='#3498db')
        ax2.axhline(y=result['confidence'] * 100, color='#2ecc71', 
                   linestyle='--', linewidth=2, label=f'Predicted: {result["confidence"]*100:.1f}%')
        ax2.set_xlabel('Category Rank', fontsize=12)
        ax2.set_ylabel('Confidence (%)', fontsize=12)
        ax2.set_title('Confidence Distribution Across All Categories', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'outputs/results/prediction_{timestamp}.png'
        os.makedirs('outputs/results', exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {filename}")
        
        plt.show()
    
    def print_result(self, result, resume_text):
        """Print prediction results in a formatted way."""
        print("\n" + "="*70)
        print(" PREDICTION RESULTS")
        print("="*70)
        
        # Resume snippet
        print(f"\n📄 Resume Text (first 200 chars):")
        print(f"   {resume_text[:200]}...")
        
        # Main prediction
        print(f"\n🎯 Predicted Category: {result['category']}")
        print(f"   Category ID: {result['category_id']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Confidence level
        if result['confidence'] > 0.9:
            confidence_level = "Very High ✨"
        elif result['confidence'] > 0.7:
            confidence_level = "High ✓"
        elif result['confidence'] > 0.5:
            confidence_level = "Moderate ~"
        else:
            confidence_level = "Low ⚠"
        print(f"   Confidence Level: {confidence_level}")
        
        # Top 5 predictions
        print(f"\n📊 Top 5 Predictions:")
        for i, (cat, score) in enumerate(result['top_predictions'], 1):
            bar = "█" * int(score * 50)
            print(f"   {i}. {cat:30s} {score:6.2%} {bar}")
        
        print("\n" + "="*70 + "\n")

def get_sample_resumes():
    """Return a collection of sample resumes for testing."""
    return {
        "Data Scientist": """
            Skills: Python, R, Machine Learning, Deep Learning, TensorFlow, PyTorch, scikit-learn, 
            pandas, NumPy, SQL, Statistical Analysis, Data Visualization, NLP, Computer Vision
            
            Experience: 4 years as Data Scientist at tech companies
            - Built predictive models achieving 95% accuracy for customer churn prediction
            - Developed recommendation systems using collaborative filtering
            - Implemented deep learning models for image classification
            - Created data pipelines for processing large datasets
            
            Projects:
            - Sentiment Analysis using BERT and transformers
            - Time series forecasting with LSTM networks
            - Customer segmentation using K-means clustering
            
            Education: Masters in Data Science, PhD in Machine Learning
            Certifications: AWS Certified Machine Learning, Google Professional Data Engineer
        """,
        
        "Python Developer": """
            Skills: Python, Django, Flask, FastAPI, REST APIs, PostgreSQL, MongoDB, Redis,
            Docker, Git, Linux, AWS, Microservices, Unit Testing, CI/CD
            
            Experience: 5 years as Python Backend Developer
            - Developed scalable REST APIs serving 10M+ requests per day
            - Built microservices architecture using FastAPI and Docker
            - Optimized database queries reducing response time by 60%
            - Implemented caching strategies with Redis
            
            Projects:
            - E-commerce platform backend with Django and PostgreSQL
            - Real-time chat application using WebSockets
            - Task scheduling system with Celery and RabbitMQ
            
            Education: Bachelor's in Computer Science
            GitHub: 500+ contributions, 50+ repositories
        """,
        
        "DevOps Engineer": """
            Skills: AWS, Azure, Docker, Kubernetes, Jenkins, GitLab CI/CD, Terraform,
            Ansible, Linux, Bash, Python, Monitoring (Prometheus, Grafana), ELK Stack
            
            Experience: 3 years in DevOps and Cloud Infrastructure
            - Managed AWS infrastructure serving 5M users
            - Implemented CI/CD pipelines reducing deployment time by 70%
            - Automated infrastructure provisioning with Terraform
            - Set up monitoring and alerting systems
            
            Achievements:
            - Reduced cloud costs by 40% through optimization
            - Achieved 99.9% uptime for production systems
            - Implemented zero-downtime deployments
            
            Certifications: AWS Solutions Architect, Certified Kubernetes Administrator
            Education: Bachelor's in Information Technology
        """,
        
        "Java Developer": """
            Skills: Java, Spring Boot, Spring Framework, Hibernate, Microservices,
            REST APIs, MySQL, Oracle, Maven, JUnit, Git, Agile/Scrum
            
            Experience: 6 years in Java Enterprise Development
            - Developed enterprise applications using Spring Boot
            - Built microservices handling 50K+ transactions per day
            - Implemented OAuth2 security and JWT authentication
            - Optimized application performance and memory usage
            
            Projects:
            - Banking system with Spring Security and Hibernate
            - Inventory management system with Spring MVC
            - Payment gateway integration with third-party APIs
            
            Education: Bachelor's in Computer Engineering
            Certifications: Oracle Certified Java Programmer
        """,
        
        "Web Developer": """
            Skills: HTML5, CSS3, JavaScript, React.js, Vue.js, Node.js, TypeScript,
            Redux, Webpack, Sass, Responsive Design, RESTful APIs, MongoDB
            
            Experience: 4 years as Full Stack Web Developer
            - Built responsive web applications using React and Redux
            - Developed RESTful APIs with Node.js and Express
            - Implemented modern UI/UX designs with Material-UI
            - Optimized web performance achieving 90+ Lighthouse scores
            
            Projects:
            - Social media platform with React and Node.js
            - E-commerce website with payment integration
            - Real-time dashboard with WebSockets and Chart.js
            
            Portfolio: www.portfolio.com with 20+ live projects
            Education: Bachelor's in Web Development
        """,
        
        "Mechanical Engineer": """
            Skills: AutoCAD, SolidWorks, CATIA, ANSYS, CAM, GD&T, Manufacturing Processes,
            Materials Science, Thermodynamics, Fluid Mechanics, 3D Modeling
            
            Experience: 5 years in Mechanical Design and Manufacturing
            - Designed mechanical components for automotive industry
            - Performed FEA analysis using ANSYS
            - Optimized manufacturing processes reducing costs by 25%
            - Collaborated with cross-functional teams on product development
            
            Projects:
            - Engine component design and optimization
            - HVAC system design for commercial buildings
            - Manufacturing automation system
            
            Education: Bachelor's in Mechanical Engineering, Masters in Design
            Certifications: Certified SolidWorks Professional, Six Sigma Green Belt
        """,
        
        "HR Professional": """
            Skills: Talent Acquisition, Employee Relations, Performance Management,
            HR Analytics, Compensation & Benefits, HRIS, Recruitment, Training & Development
            
            Experience: 6 years in Human Resources Management
            - Managed end-to-end recruitment for 200+ positions annually
            - Implemented HR policies improving employee satisfaction by 35%
            - Conducted training programs for 500+ employees
            - Reduced employee turnover by 20% through retention strategies
            
            Achievements:
            - Built employer branding strategy attracting top talent
            - Implemented performance management system
            - Managed compensation and benefits for 1000+ employees
            
            Education: MBA in Human Resource Management
            Certifications: SHRM-CP, HRCI PHR
        """
    }

def interactive_mode(classifier):
    """Interactive mode to test custom resumes."""
    print("\n" + "="*70)
    print(" INTERACTIVE MODE")
    print("="*70)
    print("\nOptions:")
    print("1. Test sample resumes")
    print("2. Enter custom resume text")
    print("3. Batch test all samples")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            samples = get_sample_resumes()
            print("\nAvailable sample resumes:")
            for i, name in enumerate(samples.keys(), 1):
                print(f"{i}. {name}")
            
            sample_choice = input("\nSelect a sample (1-{}): ".format(len(samples)))
            try:
                sample_idx = int(sample_choice) - 1
                sample_name = list(samples.keys())[sample_idx]
                resume_text = samples[sample_name]
                
                print(f"\n🔍 Testing: {sample_name}")
                result = classifier.predict(resume_text)
                classifier.print_result(result, resume_text)
                
                viz = input("Visualize results? (y/n): ").lower()
                if viz == 'y':
                    classifier.visualize_prediction(result, resume_text)
            except (ValueError, IndexError):
                print("Invalid selection!")
        
        elif choice == '2':
            print("\nEnter resume text (type 'END' on a new line to finish):")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            
            resume_text = '\n'.join(lines)
            if resume_text.strip():
                result = classifier.predict(resume_text)
                classifier.print_result(result, resume_text)
                
                viz = input("Visualize results? (y/n): ").lower()
                if viz == 'y':
                    classifier.visualize_prediction(result, resume_text)
            else:
                print("No text entered!")
        
        elif choice == '3':
            print("\n🔄 Batch testing all sample resumes...\n")
            samples = get_sample_resumes()
            
            results_summary = []
            for name, resume_text in samples.items():
                result = classifier.predict(resume_text, show_top_n=3)
                results_summary.append((name, result))
                print(f"✓ {name:25s} → {result['category']:25s} ({result['confidence']:.1%})")
            
            print("\n" + "="*70)
            print(" BATCH TEST SUMMARY")
            print("="*70)
            
            correct = 0
            for expected, result in results_summary:
                predicted = result['category']
                is_correct = expected.lower() in predicted.lower() or predicted.lower() in expected.lower()
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                print(f"{status} Expected: {expected:25s} | Predicted: {predicted:25s} ({result['confidence']:.1%})")
            
            accuracy = (correct / len(results_summary)) * 100
            print(f"\nAccuracy: {correct}/{len(results_summary)} ({accuracy:.1f}%)")
            print("="*70)
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-4.")

def main():
    """Main function to run the demo."""
    try:
        # Initialize classifier
        classifier = ResumeClassifier()
        
        # Run interactive mode
        interactive_mode(classifier)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
