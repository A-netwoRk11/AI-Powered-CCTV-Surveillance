#!/usr/bin/env python3
"""
Test script to verify deployment setup
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import config.settings as settings
        print("✅ config.settings imported successfully")
        print(f"   BASE_DIR: {settings.BASE_DIR}")
        print(f"   OUTPUT_DIR: {settings.OUTPUT_DIR}")
        print(f"   MODELS_DIR: {settings.MODELS_DIR}")
    except Exception as e:
        print(f"❌ Failed to import config.settings: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ultralytics: {e}")
        return False
    
    try:
        import cv2
        print("✅ cv2 imported successfully")
    except Exception as e:
        print(f"❌ Failed to import cv2: {e}")
        return False
    
    try:
        from flask import Flask
        print("✅ Flask imported successfully")
    except Exception as e:
        print(f"❌ Failed to import Flask: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test YOLO model initialization"""
    print("\nTesting model initialization...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✅ YOLO model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize YOLO model: {e}")
        return False

def test_directories():
    """Test directory creation"""
    print("\nTesting directory creation...")
    
    try:
        import config.settings as settings
        
        dirs_to_test = [settings.OUTPUT_DIR, settings.UPLOADS_DIR, settings.OUTPUT_VIDEOS_DIR]
        
        for directory in dirs_to_test:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created/verified directory: {directory}")
            except Exception as e:
                print(f"⚠️ Could not create directory {directory}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Directory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running deployment tests...\n")
    
    tests = [
        test_imports,
        test_model_initialization, 
        test_directories
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Deployment setup looks good.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
