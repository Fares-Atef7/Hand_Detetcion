# Hand Edge Detection and Image Matching

This project is designed to detect edges in hand images captured via a webcam or mobile camera and match them against a dataset of preprocessed images. It uses image preprocessing techniques, edge detection (Sobel), and K-Means clustering for segmentation.

## Features

- **Real-time edge detection**: Captures images from a webcam or mobile camera and detects edges using Sobel edge detection.
- **Image preprocessing**: Converts images to grayscale, applies Gaussian blur, sharpening, and Otsu's thresholding.
- **K-Means clustering**: Segments images using K-Means clustering for better visualization.
- **Template matching**: Matches detected edges against a dataset of preprocessed images.

## Installation 
you need first to install these lib ..
pip install opencv-python
pip install numpy
pip install matplotlib
