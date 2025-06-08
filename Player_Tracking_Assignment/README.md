# Player Tracking and Re-identification System

## 🎯 Project Overview

This project implements an advanced **Player Tracking and Re-identification System** for sports videos, designed to maintain consistent player IDs even when players go out of frame and reappear. The system uses YOLOv11 for object detection combined with sophisticated tracking algorithms and appearance-based re-identification techniques.

### 🔥 Key Features

- **Real-time Player Detection**: Uses fine-tuned YOLOv11 model for accurate player detection
- **Intelligent Re-identification**: Maintains consistent player IDs when players reappear in frame
- **Multi-modal Tracking**: Combines position-based and appearance-based tracking
- **Performance Optimization**: Optimized for real-time processing with detailed performance metrics
- **Comprehensive Analysis**: Includes trajectory analysis and performance visualization tools

## 🏗️ System Architecture

The system consists of several key components:

### Core Modules

1. **PlayerTracker**: Main tracking class implementing the tracking pipeline
2. **FeatureExtractor**: Extracts appearance features for re-identification
3. **Main Processing Script**: Handles video processing and result generation
4. **Analysis Tools**: Provides performance analysis and visualization

### Algorithm Pipeline

```
Input Video Frame
       ↓
Object Detection (YOLOv11)
       ↓
Feature Extraction (Color + Texture + Spatial)
       ↓
Player Matching (Hungarian Algorithm)
       ↓
Re-identification (Appearance Similarity)
       ↓
ID Assignment & Tracking Update
       ↓
Annotated Output Frame
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Windows 10/11, macOS, or Linux

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# If you encounter issues with PyTorch, install manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation

```bash
# Test the installation
python -c "import cv2, torch, ultralytics; print('✓ All dependencies installed successfully')"
```

## 📖 Usage Guide

### Basic Usage

Run the player tracking system on the provided video:

```bash
python main.py --save_video --display
```

### Advanced Usage

```bash
# Full command with all options
python main.py \
    --input_video "../input_video/15sec_input_720p.mp4" \
    --model_path "../object detection model/best.pt" \
    --output_dir "output" \
    --confidence 0.5 \
    --save_video \
    --display
```

### Command Line Arguments

- `--input_video`: Path to input video file
- `--model_path`: Path to YOLOv11 model file  
- `--output_dir`: Directory to save results
- `--confidence`: Detection confidence threshold
- `--save_video`: Save annotated output video
- `--display`: Display video during processing

### Analysis Tools

After processing, analyze the results:

```bash
python analysis.py --results_dir output --save_plots
```

## 📊 Output Files

- `tracked_output.mp4`: Annotated video with tracking overlays
- `tracking_results.json`: Performance metrics and statistics
- `detailed_tracking.pkl`: Detailed tracking data
- `tracking_report.txt`: Comprehensive analysis report
- `performance_analysis.png`: Performance visualization
- `trajectory_analysis.png`: Player trajectory plots

## 🔧 Technical Implementation

### Tracking Algorithm

#### 1. Object Detection
- Fine-tuned YOLOv11 model for player detection
- Confidence thresholding for reliable detections
- Bounding box and score extraction

#### 2. Feature Extraction
- **Color Features**: HSV histograms (170 dims)
- **Texture Features**: Gradient descriptors (86 dims)  
- **Spatial Features**: Regional statistics (48 dims)

#### 3. Player Matching
- Hungarian algorithm for optimal assignment
- Combined positional and appearance distances
- Adaptive thresholding

#### 4. Re-identification
- Appearance memory with feature history
- Cosine similarity for matching
- Temporal consistency handling

## 📈 Performance Metrics

### Accuracy Metrics
- Player re-identification accuracy
- ID consistency across frames
- Detection precision and recall

### Efficiency Metrics
- Processing FPS
- Average processing time per frame
- Memory usage
- Real-time capability

## 🚧 Challenges and Solutions

### Challenge 1: Player Occlusions
**Solution**: Robust partial-view feature extraction and temporal persistence

### Challenge 2: Similar Appearances  
**Solution**: Multi-modal features and trajectory-based disambiguation

### Challenge 3: Lighting Variations
**Solution**: HSV color space and normalized features

### Challenge 4: Real-time Performance
**Solution**: Optimized operations and efficient data structures

## 🔮 Future Improvements

- Deep learning re-identification with Siamese networks
- Kalman filters for motion prediction
- Scene understanding and court detection
- Multi-camera fusion capabilities
- Pose estimation integration

## 📚 Dependencies

### Core Libraries
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

## 🆘 Troubleshooting

### CUDA Issues
```bash
export CUDA_VISIBLE_DEVICES=""
python main.py --save_video
```

### Model Loading
Verify model file exists and is accessible

### Video Codecs
```bash
pip install opencv-python-headless
```

### Memory Issues
Use higher confidence threshold to reduce detections

## 🎉 Conclusion

This system demonstrates advanced computer vision techniques for sports video analysis, balancing accuracy, efficiency, and reliability for both research and practical applications. 