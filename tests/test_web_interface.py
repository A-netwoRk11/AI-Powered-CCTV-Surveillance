<<<<<<< HEAD
#!/usr/bin/env python3
import requests
import sys
import os

def test_web_interface():
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

def test_upload_endpoint():
    try:
        print("\n📤 Testing upload endpoint...")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(base_dir, "input", "demo_videos", "dogwithBall.mp4")
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
            
        print(f"📁 Using test file: {test_file}")
        
        with open(test_file, 'rb') as f:
            files = {'video': ('dogwithBall.mp4', f, 'video/mp4')}
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

if __name__ == "__main__":
    print("🧪 Testing AI-Powered CCTV Surveillance Web Interface")
    print("=" * 60)
    
    web_ok = test_web_interface()
    
    if web_ok:
        upload_ok = test_upload_endpoint()
        
        if upload_ok:
            print("\n🎉 All tests passed! Web interface is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Upload test failed!")
            sys.exit(1)
    else:
        print("\n❌ Web interface test failed!")
        sys.exit(1)
=======
#!/usr/bin/env python3
import requests
import sys
import os

def test_web_interface():
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

def test_upload_endpoint():
    try:
        print("\n📤 Testing upload endpoint...")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(base_dir, "input", "demo_videos", "dogwithBall.mp4")
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
            
        print(f"📁 Using test file: {test_file}")
        
        with open(test_file, 'rb') as f:
            files = {'video': ('dogwithBall.mp4', f, 'video/mp4')}
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

if __name__ == "__main__":
    print("🧪 Testing AI-Powered CCTV Surveillance Web Interface")
    print("=" * 60)
    
    web_ok = test_web_interface()
    
    if web_ok:
        upload_ok = test_upload_endpoint()
        
        if upload_ok:
            print("\n🎉 All tests passed! Web interface is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Upload test failed!")
            sys.exit(1)
    else:
        print("\n❌ Web interface test failed!")
        sys.exit(1)
>>>>>>> 91276018eae2976736e3a9f79ca130b98400e8fb
