"""
Test script for Resume Analyser API
Tests all endpoints with sample data
"""

import requests
import json
import time

# API URL
API_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_root():
    """Test root endpoint"""
    print_section("Testing Root Endpoint")
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_categories():
    """Test categories endpoint"""
    print_section("Testing Categories Endpoint")
    response = requests.get(f"{API_URL}/categories")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total Categories: {data['total']}")
    print(f"Sample Categories: {list(data['categories'].values())[:5]}...")
    return response.status_code == 200

def test_single_prediction():
    """Test single resume prediction"""
    print_section("Testing Single Prediction")
    
    resume_text = """
    Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow, NLP, Computer Vision
    Experience: 3 years in Data Science and AI model development
    Projects: Built recommendation systems, sentiment analysis models, image classification
    Education: Masters in Computer Science with specialization in Machine Learning
    Achievements: Published 2 research papers in ML conferences
    """
    
    print(f"Resume Text: {resume_text[:100]}...")
    
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": resume_text}
    )
    end_time = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Category: {result['category']}")
        print(f"  Category ID: {result['category_id']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_single_prediction_with_scores():
    """Test single prediction with all scores"""
    print_section("Testing Single Prediction (with all scores)")
    
    resume_text = """
    Java Developer with 5 years of experience in Spring Boot, Microservices, REST APIs
    Expert in Java 8/11, Spring Framework, Hibernate, MySQL, Docker, Kubernetes
    Led development of scalable backend systems handling 1M+ requests per day
    """
    
    print(f"Resume Text: {resume_text[:100]}...")
    
    response = requests.post(
        f"{API_URL}/predict?include_all_scores=true",
        json={"text": resume_text}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Category: {result['category']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        # Show top 5 scores
        if result['all_scores']:
            sorted_scores = sorted(
                result['all_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print(f"\n  Top 5 Categories:")
            for cat, score in sorted_scores:
                print(f"    {cat}: {score:.2%}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_batch_prediction():
    """Test batch resume prediction"""
    print_section("Testing Batch Prediction")
    
    resumes = [
        "Python developer with 5 years experience in Django, Flask, REST APIs, PostgreSQL",
        "Data scientist with expertise in machine learning, deep learning, statistics, Python, R",
        "DevOps engineer experienced in AWS, Docker, Kubernetes, CI/CD, Jenkins, Terraform",
        "Web developer skilled in React, JavaScript, HTML/CSS, Node.js, MongoDB",
        "Mechanical engineer with 4 years in CAD design, manufacturing, AutoCAD, SolidWorks"
    ]
    
    print(f"Testing with {len(resumes)} resumes")
    
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/batch-predict",
        json={"resumes": resumes}
    )
    end_time = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Total Inference Time: {(end_time - start_time)*1000:.2f} ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Processed: {result['total_processed']}")
        print(f"Average Time per Resume: {((end_time - start_time)*1000)/len(resumes):.2f} ms")
        
        print(f"\nPredictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['category']} (confidence: {pred['confidence']:.2%})")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_error_handling():
    """Test error handling with invalid input"""
    print_section("Testing Error Handling")
    
    # Test with too short text
    print("Test 1: Too short text")
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": "short"}
    )
    print(f"Status Code: {response.status_code} (Expected: 422)")
    
    # Test with empty batch
    print("\nTest 2: Empty batch")
    response = requests.post(
        f"{API_URL}/batch-predict",
        json={"resumes": []}
    )
    print(f"Status Code: {response.status_code} (Expected: 422)")
    
    return True

def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("  RESUME ANALYSER API - TEST SUITE")
    print("="*60)
    print(f"API URL: {API_URL}")
    print(f"Testing started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Categories", test_categories),
        ("Single Prediction", test_single_prediction),
        ("Single Prediction with Scores", test_single_prediction_with_scores),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nError in {name}: {str(e)}")
            results.append((name, False))
        time.sleep(0.5)  # Small delay between tests
    
    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running at http://localhost:8000")
        print("\nStart the API with:")
        print("  python app.py")
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
