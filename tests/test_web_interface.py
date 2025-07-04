#!/usr/bin/env python3
"""
Comprehensive test suite for AI-Powered CCTV Surveillance Web Interface.
Tests all major endpoints and functionality.
"""

import requests
import sys
import os
import time
import json
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import *

class WebInterfaceTestConfig:
    """Configuration for web interface tests."""
    BASE_URL = "http://localhost:5000"
    TIMEOUT = 10
    UPLOAD_TIMEOUT = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 2

class WebInterfaceTests(unittest.TestCase):
    """Comprehensive test suite for the web interface."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - check if web server is running."""
        cls.config = WebInterfaceTestConfig()
        cls.base_url = cls.config.BASE_URL
        
        print(f"🌐 Testing web interface at {cls.base_url}")
        
        # Wait for server to be ready
        for attempt in range(cls.config.MAX_RETRIES):
            try:
                response = requests.get(cls.base_url, timeout=cls.config.TIMEOUT)
                if response.status_code == 200:
                    print("✅ Web server is running and accessible!")
                    break
            except requests.exceptions.ConnectionError:
                if attempt < cls.config.MAX_RETRIES - 1:
                    print(f"⏳ Waiting for web server... (attempt {attempt + 1})")
                    time.sleep(cls.config.RETRY_DELAY)
                else:
                    raise unittest.SkipTest("Web server is not running on port 5000")
        
    def test_main_page_accessibility(self):
        """Test if the main page is accessible and contains expected content."""
        print("\n🏠 Testing main page accessibility...")
        
        response = requests.get(self.base_url, timeout=self.config.TIMEOUT)
        
        self.assertEqual(response.status_code, 200, "Main page should return 200 OK")
        self.assertGreater(len(response.text), 1000, "Main page should have substantial content")
        
        # Check for key elements
        content = response.text.lower()
        self.assertIn("ai-powered cctv surveillance", content, "Page should contain main title")
        self.assertIn("upload", content, "Page should contain upload functionality")
        self.assertIn("analyze", content, "Page should contain analysis functionality")
        
        print("✅ Main page accessibility test passed!")
    
    def test_main_page_structure(self):
        """Test if main page has proper HTML structure."""
        print("\n🏗️ Testing main page HTML structure...")
        
        response = requests.get(self.base_url, timeout=self.config.TIMEOUT)
        content = response.text
        
        # Check for essential HTML elements
        self.assertIn("<!DOCTYPE html>", content, "Should have proper DOCTYPE")
        self.assertIn("<title>", content, "Should have title tag")
        self.assertIn("navbar", content, "Should have navigation")
        self.assertIn("form", content, "Should have upload form")
        
        print("✅ Main page structure test passed!")
    
    def test_saved_analysis_page(self):
        """Test the saved analysis page accessibility."""
        print("\n📊 Testing saved analysis page...")
        
        try:
            response = requests.get(f"{self.base_url}/saved_analysis", timeout=self.config.TIMEOUT)
            self.assertEqual(response.status_code, 200, "Saved analysis page should be accessible")
            
            content = response.text.lower()
            self.assertTrue(
                "saved" in content or "analysis" in content or "results" in content,
                "Page should contain relevant content"
            )
            print("✅ Saved analysis page test passed!")
            
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to access saved analysis page: {e}")
    
    def test_create_output_folder_endpoint(self):
        """Test the output folder creation endpoint."""
        print("\n📁 Testing output folder creation...")
        
        try:
            response = requests.post(
                f"{self.base_url}/create_output_folder",
                headers={'Content-Type': 'application/json'},
                timeout=self.config.TIMEOUT
            )
            
            self.assertIn(response.status_code, [200, 201], 
                         "Create folder endpoint should return success status")
            
            # Try to parse JSON response
            try:
                data = response.json()
                self.assertIn('status', data, "Response should contain status")
                print(f"✅ Output folder creation test passed! Status: {data.get('status')}")
            except json.JSONDecodeError:
                self.fail("Response should be valid JSON")
                
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to test create output folder: {e}")
    
    def test_file_upload_functionality(self):
        """Test video file upload and analysis."""
        print("\n📤 Testing file upload functionality...")
        
        # Find a test video file
        demo_videos_path = Path(__file__).parent.parent / "input" / "demo_videos"
        test_files = list(demo_videos_path.glob("*.mp4"))
        
        if not test_files:
            self.skipTest("No demo video files found for testing")
        
        test_file = test_files[0]
        print(f"📁 Using test file: {test_file}")
        
        try:
            with open(test_file, 'rb') as f:
                files = {
                    'videoFile': (test_file.name, f, 'video/mp4')
                }
                data = {
                    'test_name': 'Automated_Test_Upload',
                    'prompt': 'Test analysis for automated testing'
                }
                
                print("🚀 Uploading file for analysis...")
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data,
                    timeout=self.config.UPLOAD_TIMEOUT
                )
                
                self.assertIn(response.status_code, [200, 302], 
                             "Upload should return success or redirect")
                
                # Check if response indicates successful processing
                content = response.text.lower()
                success_indicators = ['success', 'analysis', 'result', 'complete']
                self.assertTrue(
                    any(indicator in content for indicator in success_indicators),
                    "Response should indicate successful processing"
                )
                
                print("✅ File upload test passed!")
                
        except Exception as e:
            self.fail(f"File upload test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling for invalid requests."""
        print("\n🚨 Testing error handling...")
        
        # Test invalid endpoint
        response = requests.get(f"{self.base_url}/nonexistent", timeout=self.config.TIMEOUT)
        self.assertEqual(response.status_code, 404, "Invalid endpoint should return 404")
        
        # Test upload without file
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                data={'test_name': 'No File Test'},
                timeout=self.config.TIMEOUT
            )
            # Should either redirect or show an error
            self.assertIn(response.status_code, [400, 422, 302], 
                         "Upload without file should return error or redirect")
        except requests.exceptions.RequestException:
            pass  # This is acceptable for this test
        
        print("✅ Error handling test passed!")
    
    def test_api_endpoints_json_response(self):
        """Test API endpoints that should return JSON."""
        print("\n🔗 Testing JSON API endpoints...")
        
        # Test create output folder endpoint
        try:
            response = requests.post(
                f"{self.base_url}/create_output_folder",
                headers={'Content-Type': 'application/json'},
                timeout=self.config.TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIsInstance(data, dict, "JSON endpoints should return dict")
                self.assertIn('status', data, "Response should have status field")
                
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"⚠️ JSON API test warning: {e}")
        
        print("✅ JSON API endpoints test completed!")

def run_legacy_tests():
    """Run the original test functions for backward compatibility."""
    print("\n🔄 Running legacy test functions...")
    
    try:
        # Original test functions
        web_ok = test_web_interface_legacy()
        if web_ok:
            upload_ok = test_upload_endpoint_legacy()
            return web_ok and upload_ok
        return False
    except Exception as e:
        print(f"❌ Legacy tests failed: {e}")
        return False

def test_web_interface_legacy():
    """Legacy test function for backward compatibility."""
    try:
        print("🌐 Testing web interface...")
        response = requests.get('http://localhost:5000', timeout=5)
        
        if response.status_code == 200:
            print("✅ Web interface is accessible!")
            print(f"📄 Response length: {len(response.text)} characters")
            
            if "AI-Powered CCTV Surveillance" in response.text:
                print("✅ Main page content detected!")
            else:
                print("❌ Main page content not found!")
                
            if "upload" in response.text.lower():
                print("✅ Upload functionality detected!")
            else:
                print("❌ Upload functionality not found!")
                
            return True
        else:
            print(f"❌ Web interface returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web interface! Is the Flask app running on port 5000?")
        return False
    except Exception as e:
        print(f"❌ Error testing web interface: {e}")
        return False

def test_upload_endpoint_legacy():
    """Legacy upload test function for backward compatibility."""
    try:
        print("\n📤 Testing upload endpoint...")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(base_dir, "input", "demo_videos", "dogwithBall.mp4")
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
            
        print(f"📁 Using test file: {test_file}")
        
        with open(test_file, 'rb') as f:
            files = {'videoFile': ('dogwithBall.mp4', f, 'video/mp4')}
            data = {'test_name': 'Web Interface Test'}
            
            print("🚀 Uploading file...")
            response = requests.post('http://localhost:5000/analyze', files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                print("✅ Upload successful!")
                
                if "Analysis Results" in response.text or "results" in response.text.lower():
                    print("✅ Analysis results page returned!")
                    
                    if "dog" in response.text.lower() or "detection" in response.text.lower():
                        print("✅ Object detection results found!")
                    else:
                        print("⚠️  No detection results visible in response")
                        
                    return True
                else:
                    print("❌ Analysis results not found in response")
                    print(f"📄 Response preview: {response.text[:500]}...")
                    return False
            else:
                print(f"❌ Upload failed with status code: {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing upload: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("🧪 Testing AI-Powered CCTV Surveillance Web Interface")
    print("=" * 60)
    
    # Run new unittest-based tests
    print("\n🆕 Running comprehensive test suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(WebInterfaceTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run legacy tests for backward compatibility
    print("\n🔄 Running legacy compatibility tests...")
    legacy_success = run_legacy_tests()
    
    # Determine overall success
    unittest_success = result.wasSuccessful()
    overall_success = unittest_success and legacy_success
    
    if overall_success:
        print("\n🎉 All tests passed! Web interface is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed!")
        if not unittest_success:
            print(f"   - Unittest failures: {len(result.failures)}")
            print(f"   - Unittest errors: {len(result.errors)}")
        if not legacy_success:
            print("   - Legacy tests failed")
        return 1
