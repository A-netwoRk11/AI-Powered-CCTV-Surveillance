## AI-powered CCTV surveillance system for Industrial monitoring.
## 🚀 Quick Start

## Step-by-Step Installation

### 1. Check Python Version
```bash
python --version
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 5. Install Requirements (Choose based on Python version)

**For Python 3.8-3.11:**
```bash
pip install -r requirements.txt
```

**For Python 3.12:**
```bash
pip install -r requirements-python312.txt
```

### Minimal Installation (Manual - Core packages only)
**For Python 3.8-3.11:**
```bash
pip install Flask==2.3.3 gunicorn==21.2.0 opencv-python-headless==4.8.1.78 ultralytics>=8.0.0,<8.1.0 torch>=2.0.0,<2.2.0 numpy>=1.24.3,<1.26.0 PyYAML>=6.0.1 requests>=2.31.0 tqdm>=4.64.0
```
**For Python 3.12:**
```bash
pip install Flask==3.0.0 gunicorn==21.2.0 opencv-python-headless==4.9.0.80 ultralytics>=8.1.0 torch>=2.2.0 numpy>=1.26.0 PyYAML>=6.0.1 requests>=2.31.0 tqdm>=4.66.0
```

### Method 2: Manual Installation by Python Version

**For Python 3.8-3.11:**
```bash
pip install Flask==2.3.3 Werkzeug==2.3.7 gunicorn==21.2.0
pip install opencv-python-headless==4.8.1.78
pip install ultralytics>=8.0.0,<8.1.0 Pillow>=10.0.1
pip install torch>=2.0.0,<2.2.0 torchvision>=0.15.0,<0.17.0
pip install numpy>=1.24.3,<1.26.0 PyYAML>=6.0.1 requests>=2.31.0
```

**For Python 3.12:**
```bash
pip install Flask==3.0.0 Werkzeug==3.0.1 gunicorn==21.2.0
pip install opencv-python-headless==4.9.0.80
pip install ultralytics>=8.1.0 Pillow>=10.2.0
pip install torch>=2.2.0 torchvision>=0.17.0
pip install numpy>=1.26.0 PyYAML>=6.0.1 requests>=2.31.0
```

## start webapp with below code
```bash
python src/main.py
```

This launches the modern web interface at `http://localhost:5000` with all surveillance features accessible through your browser.

## 🌐 Web Interface Features

### Browser-Based Control
- **Video Upload & Analysis**: Upload surveillance videos for AI processing
- **Modern UI**: Clean, responsive interface for viewing results
- **Remote Access**: Access from any device on your network
- **Webcam Control**: Start/stop real-time webcam surveillance
- **Output Management**: View and manage analysis results and screenshots



## 🚀 Deploy to Render (Production)

This project is ready for deployment on Render with the included configuration files.

### Prerequisites
- GitHub account
- Render account (free tier available)

### Step-by-Step Deployment Guide

#### 1. Prepare Your GitHub Repository
```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - AI CCTV Surveillance System"

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/ai-cctv-surveillance.git

# Push to GitHub
git push -u origin main
```

#### 2. Deploy on Render

1. **Sign up/Login to Render**: Go to [render.com](https://render.com) and sign up or login
2. **Connect GitHub**: Connect your GitHub account to Render
3. **Create New Web Service**: 
   - Click "New +" → "Web Service"
   - Select "Build and deploy from a Git repository"
   - Connect your GitHub repository

#### 3. Configure Render Settings

**Basic Settings:**
- **Name**: `ai-cctv-surveillance` (or your preferred name)
- **Region**: Choose closest to your location
- **Branch**: `main`
- **Runtime**: `Python 3`

**Build & Deploy Settings:**
- **Build Command**: `./build.sh`
- **Start Command**: `./start.sh`

**Advanced Settings:**
- **Auto-Deploy**: `Yes` (recommended)

#### 4. Environment Variables (Optional)
If you need custom settings, add these in Render's Environment tab:
- `FLASK_ENV`: `production`
- `PORT`: `10000` (Render will set this automatically)

#### 5. Deploy
1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies using `build.sh`
   - Start the application using `start.sh`
3. Your app will be available at: `https://your-app-name.onrender.com`

### Deployment Files Included

This project includes all necessary deployment files:

- **`runtime.txt`**: Specifies Python 3.11 for Render
- **`requirements.txt`**: Core dependencies for Python 3.8-3.11
- **`requirements-python312.txt`**: Dependencies for Python 3.12
- **`build.sh`**: Installation script for Render
- **`start.sh`**: Startup script with Gunicorn
- **`.gitignore`**: Excludes unnecessary files from deployment

### Post-Deployment Notes

1. **Free Tier Limitations**: Render's free tier may have some limitations:
   - Services sleep after 15 minutes of inactivity
   - Limited build minutes per month
   - Slower cold starts

2. **Upgrade for Production**: For production use, consider Render's paid plans for:
   - Always-on services
   - Faster performance
   - More resources

3. **Custom Domain**: You can add a custom domain in Render's settings

### Troubleshooting Deployment

If deployment fails:

1. **Check Logs**: View build and runtime logs in Render dashboard
2. **Python Version**: Ensure `runtime.txt` matches your requirements
3. **Dependencies**: Verify all packages install correctly
4. **File Permissions**: Ensure `build.sh` and `start.sh` are executable

### Local Development vs Production

**Local Development:**
```bash
python src/main.py
```

**Production (Render):**
- Uses Gunicorn WSGI server
- Environment variables for configuration
- Optimized for production workloads