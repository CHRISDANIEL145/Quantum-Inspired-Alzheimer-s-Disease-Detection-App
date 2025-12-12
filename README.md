# 🧠 Quantum-Inspired Alzheimer's Disease Detection App

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced deep learning web application for early detection of Alzheimer's Disease from brain MRI scans. This project leverages a VGG16-based convolutional neural network to classify brain scans into four stages of cognitive decline with high accuracy.

![NeuroScan AI Interface](static/background.png)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Classification Categories](#-classification-categories)
- [Project Architecture](#-project-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🎯 Overview

Alzheimer's Disease is a progressive neurological disorder that causes brain cells to degenerate and die, leading to a continuous decline in thinking, behavioral, and social skills. Early detection is crucial for:

- **Better treatment outcomes** - Early intervention can slow disease progression
- **Care planning** - Families can prepare for future care needs
- **Clinical trials** - Patients may qualify for experimental treatments
- **Quality of life** - Early lifestyle changes can help manage symptoms

This application uses state-of-the-art deep learning techniques to analyze brain MRI scans and provide instant classification results with confidence scores.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **AI-Powered Analysis** | VGG16-based neural network trained on thousands of brain MRI images |
| ⚡ **Real-time Results** | Get instant predictions with confidence percentages |
| 🎨 **Modern UI** | Beautiful, responsive interface with glassmorphism design |
| 📱 **Mobile Friendly** | Fully responsive design works on all devices |
| 🔒 **Secure Upload** | Files are validated, securely processed, and automatically deleted |
| 📊 **Detailed Reports** | Comprehensive information about each classification stage |
| 🤖 **Interactive Chatbot** | Q-AIssist provides detailed information about symptoms, causes, and treatments |

---

## 🏥 Classification Categories

The model classifies brain MRI scans into four categories:

### 1. Non-Demented (Healthy)
- **Description**: No signs of cognitive decline
- **Characteristics**: Normal memory, judgment, and reasoning abilities
- **Brain Status**: No evidence of Alzheimer's-related pathological changes

### 2. Very Mild Demented
- **Description**: Earliest stage of cognitive decline
- **Characteristics**: Minor memory lapses, forgetting familiar words or object locations
- **Brain Status**: Initial formation of amyloid plaques may be starting

### 3. Mild Demented
- **Description**: Noticeable cognitive decline affecting daily activities
- **Characteristics**: Significant memory loss, getting lost in familiar places, difficulty with complex tasks
- **Brain Status**: Widespread buildup of plaques and tangles causing brain cell damage

### 4. Moderate Demented
- **Description**: Significant cognitive decline requiring assistance
- **Characteristics**: Major memory gaps, confusion about time/place, personality changes
- **Brain Status**: Extensive brain atrophy visible on MRI scans

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Upload Zone   │  │  Preview Panel  │  │  Results View   │ │
│  │  (Drag & Drop)  │  │   (Image View)  │  │   (Chatbot)     │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK WEB SERVER                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  File Validation│  │   Preprocessing │  │  Response Gen   │ │
│  │  & Security     │  │   Pipeline      │  │  & Templating   │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING ENGINE                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Image Resize   │  │   VGG16 Model   │  │  Softmax        │ │
│  │  (224x224 RGB)  │  │   Inference     │  │  Classification │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **Flask 2.x** | Web framework for API and routing |
| **TensorFlow/Keras** | Deep learning framework |
| **NumPy** | Numerical computations |
| **Pillow (PIL)** | Image processing |
| **Werkzeug** | WSGI utilities and security |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Page structure |
| **Tailwind CSS** | Utility-first styling |
| **JavaScript (ES6+)** | Interactive functionality |
| **Google Fonts** | Typography (Orbitron, Inter) |
| **Material Icons** | UI icons |

### Model Architecture
| Component | Specification |
|-----------|---------------|
| **Base Model** | VGG16 (pretrained on ImageNet) |
| **Input Shape** | 224 × 224 × 3 (RGB) |
| **Output Classes** | 4 (classification categories) |
| **Preprocessing** | VGG16 preprocess_input normalization |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Git LFS (for large model files)

### Step 1: Clone the Repository

```bash
git clone https://github.com/CHRISDANIEL145/Quantum-Inspired-Alzheimer-s-Disease-Detection-App.git
cd Quantum-Inspired-Alzheimer-s-Disease-Detection-App
```

### Step 2: Install Git LFS and Pull Large Files

```bash
# Install Git LFS (if not already installed)
git lfs install

# Pull the large model files
git lfs pull
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install flask tensorflow numpy pillow werkzeug
```

Or create a requirements.txt and install:
```bash
pip install -r requirements.txt
```

### Step 5: Verify Model File

Ensure `model.keras` exists in the root directory. This file contains the trained VGG16 model weights.

---

## 🚀 Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

### Using the Web Interface

1. **Open Browser**: Navigate to `http://127.0.0.1:5000`

2. **Upload Image**: 
   - Click the upload zone or drag & drop a brain MRI image
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: 10MB

3. **Analyze**: Click "Analyze Scan" to process the image

4. **View Results**: 
   - See the classification result and confidence score
   - Use the interactive chatbot to learn more about:
     - Overview of the condition
     - Symptoms
     - Causes
     - Diagnosis methods
     - Treatment options
     - Clinical trials

### Example Workflow

```
1. User uploads brain MRI scan (e.g., mri_scan.jpg)
           ↓
2. Image is validated (type, size, format)
           ↓
3. Image is preprocessed:
   - Converted to RGB
   - Resized to 224×224 pixels
   - Normalized using VGG16 preprocessing
           ↓
4. Model performs inference
           ↓
5. Softmax output provides class probabilities
           ↓
6. Results displayed with confidence percentage
           ↓
7. User can explore detailed information via chatbot
```

---

## 🧬 Model Details

### Architecture Overview

The model is based on **VGG16**, a deep convolutional neural network known for its excellent performance in image classification tasks.

```
Input Layer (224×224×3)
        ↓
┌───────────────────────┐
│   VGG16 Base Model    │
│   (Pretrained on      │
│    ImageNet)          │
│                       │
│   - 13 Conv Layers    │
│   - 5 MaxPool Layers  │
│   - 3 FC Layers       │
└───────────────────────┘
        ↓
    Flatten Layer
        ↓
    Dense (256, ReLU)
        ↓
    Dropout (0.5)
        ↓
    Dense (4, Softmax)
        ↓
Output: [MildDemented, ModerateDemented, NonDemented, VeryMildDemented]
```

### Training Details

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Batch Size** | 32 |
| **Input Size** | 224 × 224 × 3 |
| **Data Augmentation** | Rotation, Flip, Zoom, Shift |
| **Transfer Learning** | VGG16 pretrained weights |

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | ~95%+ |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | High |

---

## 📊 Dataset

The model was trained on the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** style dataset containing brain MRI scans categorized into four classes:

| Class | Description | Sample Count |
|-------|-------------|--------------|
| NonDemented | Healthy brain scans | ~3,200 |
| VeryMildDemented | Early-stage decline | ~2,240 |
| MildDemented | Moderate decline | ~896 |
| ModerateDemented | Significant decline | ~64 |

### Data Preprocessing Pipeline

```python
def preprocess_image(image_path):
    # 1. Load image
    img = Image.open(image_path).convert('RGB')
    
    # 2. Resize to model input size
    img = img.resize((224, 224))
    
    # 3. Convert to numpy array
    img_array = np.array(img)
    
    # 4. Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Apply VGG16 preprocessing
    img_array = preprocess_input(img_array)
    
    return img_array
```

---

## 📡 API Reference

### Endpoints

#### `GET /`
Returns the main upload page.

**Response**: HTML page with upload interface

---

#### `POST /predict`
Processes an uploaded image and returns prediction results.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Brain MRI image

**Response**: HTML page with:
- `prediction`: Classification result
- `confidence`: Confidence percentage
- `details`: Detailed information object

**Example using cURL**:
```bash
curl -X POST -F "image=@brain_scan.jpg" http://127.0.0.1:5000/predict
```

---

## 📁 Project Structure

```
Quantum-Inspired-Alzheimer-s-Disease-Detection-App/
│
├── 📄 app.py                    # Main Flask application
├── 📄 model.keras               # Trained VGG16 model (LFS)
├── 📄 .gitignore                # Git ignore rules
├── 📄 .gitattributes            # Git LFS configuration
├── 📄 README.md                 # Project documentation
│
├── 📁 static/                   # Static assets
│   ├── 🖼️ background.png        # Background image
│   └── 🎨 style.css             # Custom styles for result page
│
├── 📁 templates/                # HTML templates
│   ├── 📄 index.html            # Main upload page
│   └── 📄 result.html           # Results display page
│
├── 📁 uploads/                  # Temporary upload directory
│   └── (uploaded files - auto-deleted)
│
├── 📁 combined_images/          # Sample dataset images
│
├── 📁 ragul-early detection.../  # Additional resources
│   ├── 🖼️ background.png
│   ├── 📁 combined_images/      # Training dataset
│   ├── 📄 model_VGG16_final.keras
│   ├── 📄 quantum_inspired_alzheimers_model.keras
│   └── 📄 quantum_vgg16_alzheimer_gpu.keras
│
└── 📁 venv/                     # Virtual environment (not tracked)
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | production | Flask environment mode |
| `UPLOAD_FOLDER` | uploads | Directory for temporary uploads |
| `MAX_CONTENT_LENGTH` | 10MB | Maximum upload file size |

### Security Features

- ✅ Secure filename handling with UUID generation
- ✅ File type validation (JPG, JPEG, PNG only)
- ✅ File size limits (10MB maximum)
- ✅ Automatic file cleanup after prediction
- ✅ Debug mode disabled in production

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Step 1: Fork the Repository
Click the "Fork" button on GitHub

### Step 2: Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/Quantum-Inspired-Alzheimer-s-Disease-Detection-App.git
```

### Step 3: Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### Step 4: Make Changes
- Write clean, documented code
- Follow existing code style
- Add tests if applicable

### Step 5: Commit Changes
```bash
git add .
git commit -m "Add: your feature description"
```

### Step 6: Push and Create PR
```bash
git push origin feature/your-feature-name
```
Then create a Pull Request on GitHub.

---

## ⚠️ Disclaimer

> **IMPORTANT**: This application is for **educational and research purposes only**. It is **NOT** a substitute for professional medical diagnosis, advice, or treatment.

- Always consult qualified healthcare professionals for medical concerns
- Do not make medical decisions based solely on this tool's output
- The model's predictions should be verified by medical experts
- This tool is not FDA approved or clinically validated

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**CHRIS DANIEL**

- GitHub: [@CHRISDANIEL145](https://github.com/CHRISDANIEL145)

---

## 🙏 Acknowledgments

- **TensorFlow/Keras Team** - For the excellent deep learning framework
- **VGG Research Group** - For the VGG16 architecture
- **ADNI** - For Alzheimer's disease research and datasets
- **Flask Community** - For the lightweight web framework
- **Tailwind CSS** - For the utility-first CSS framework

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/CHRISDANIEL145/Quantum-Inspired-Alzheimer-s-Disease-Detection-App/issues) page
2. Create a new issue with detailed description
3. Include error messages and screenshots if applicable

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ for Alzheimer's Disease Research

</div>
