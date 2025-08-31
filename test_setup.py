#!/usr/bin/env python3
"""
LeafSense Setup Test Script
Tests all components to ensure proper configuration
"""

import os
import sys
import json
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'flask',
        'tensorflow',
        'opencv-python',
        'PIL',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'requests',
        'google.generativeai',
        'dotenv'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'google.generativeai':
                importlib.import_module('google.generativeai')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_file_structure():
    """Test if all required files and directories exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'train.py',
        'train.ipynb',
        'requirements.txt',
        'diseases.json',
        'README.md'
    ]
    
    required_dirs = [
        'data',
        'data/train',
        'data/val', 
        'data/test',
        'saved_models',
        'static',
        'static/css',
        'static/js',
        'static/images',
        'templates'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("✅ All files and directories present!")
        return False

def test_flask_app():
    """Test if Flask app can be imported and configured"""
    print("\n🌐 Testing Flask application...")
    
    try:
        # Temporarily modify sys.path to import app
        sys.path.insert(0, os.getcwd())
        
        # Test basic Flask import
        from flask import Flask
        print("  ✅ Flask imported successfully")
        
        # Test app.py import (without running it)
        try:
            import app
            print("  ✅ app.py imported successfully")
            
            # Check if app has required attributes
            if hasattr(app, 'app') and isinstance(app.app, Flask):
                print("  ✅ Flask app instance found")
            else:
                print("  ❌ Flask app instance not found")
                return False
                
        except Exception as e:
            print(f"  ❌ Error importing app.py: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Flask test failed: {e}")
        return False
    
    print("✅ Flask application test passed!")
    return True

def test_knowledge_base():
    """Test if diseases.json is valid and contains expected data"""
    print("\n📚 Testing knowledge base...")
    
    try:
        with open('diseases.json', 'r') as f:
            diseases = json.load(f)
        
        if not isinstance(diseases, dict):
            print("  ❌ diseases.json is not a valid JSON object")
            return False
        
        disease_count = len(diseases)
        print(f"  ✅ Found {disease_count} diseases in knowledge base")
        
        # Check structure of first disease
        first_disease = list(diseases.keys())[0]
        disease_data = diseases[first_disease]
        
        if 'description' in disease_data and 'remedy' in disease_data:
            print("  ✅ Disease data structure is correct")
        else:
            print("  ❌ Disease data structure is incorrect")
            return False
            
        # Check for some expected diseases
        expected_diseases = ['Tomato___Late_blight', 'Tomato___Early_blight', 'Potato___Late_blight']
        found_expected = [d for d in expected_diseases if d in diseases]
        
        if found_expected:
            print(f"  ✅ Found expected diseases: {', '.join(found_expected)}")
        else:
            print("  ⚠️  Some expected diseases not found")
            
    except FileNotFoundError:
        print("  ❌ diseases.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON in diseases.json: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error testing knowledge base: {e}")
        return False
    
    print("✅ Knowledge base test passed!")
    return True

def test_model_training_script():
    """Test if training script can be imported"""
    print("\n🧠 Testing model training script...")
    
    try:
        import train
        print("  ✅ train.py imported successfully")
        
        # Check if trainer class exists
        if hasattr(train, 'LeafSenseTrainer'):
            print("  ✅ LeafSenseTrainer class found")
        else:
            print("  ❌ LeafSenseTrainer class not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Error importing train.py: {e}")
        return False
    
    print("✅ Model training script test passed!")
    return True

def test_environment():
    """Test environment configuration"""
    print("\n⚙️  Testing environment configuration...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"  ✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ❌ Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+")
        return False
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("  ✅ .env file found")
    else:
        print("  ⚠️  .env file not found (you may need to create one)")
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"  ✅ Working directory: {cwd}")
    
    print("✅ Environment test passed!")
    return True

def main():
    """Run all tests"""
    print("🚀 LeafSense Setup Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_file_structure,
        test_flask_app,
        test_knowledge_base,
        test_model_training_script,
        test_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LeafSense is ready to use.")
        print("\nNext steps:")
        print("1. Set up your API keys in .env file")
        print("2. Prepare your dataset in the data/ directory")
        print("3. Run 'python train.py' to train the model")
        print("4. Run 'python app.py' to start the web application")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Create missing directories")
        print("- Check file permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
