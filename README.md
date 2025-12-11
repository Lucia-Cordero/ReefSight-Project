# Project Name
**ReefSight Project**

**Description**
This project, ReefSight, develops a Multi-Modal Deep Learning solution to predict coral bleaching risk and to classify images as 'Bleached' or 'Unbleached'. We combine image classification (VGG16) with structured environmental data (tabular analysis) to create a robust, site-specific prediction tool.

**Key Features**
|**Multi-Modal Data Integration**|: Fusion of high-resolution coral imagery with global tabular bleaching data.

|**Image Classification Pipeline**|: Automated download, cleaning, and splitting of raw image data; implementation of Baseline CNN and VGG16 Transfer Learning.

|**Tabular Data Analysis**|: Use of structured data from the Global Coral Reef Monitoring Network (GCRMN) dataset to identify environmental risk factors.

|**Performance Comparison**|: Direct comparison between image-only, tabular-only, and the final Multi-Modal Fusion Model to determine the optimal solution. develops deep learning models to automatically classify coral images as either 'Bleached' or 'Unbleached'. By leveraging computer vision and transfer learning, we aim to provide a reliable, scalable, and non-invasive tool for monitoring coral reef health, accelerating conservation efforts.

**Data used**
For the Image Classifier
|**Source** | Kaggle Dataset: Bleached Corals Detection |
| **URL** | [https://www.kaggle.com/datasets/sonainjamil/bleached-corals-detection]|
| **Classes** | 2 (Bleached, Unbleached) |
| **Format** | JPG images |
| **Preprocessing** | Images are resized to **224x224** pixels and rescaled to $[0, 1]$ before model input. |
| **Splitting Ratio**| Data is automatically split into **80% Training, 10% Validation, 10% Testing** sets. |

For the Tabular
|**Source** | Kaggle Dataset: Coral Reef Global Bleaching|
| **URL** | [https://www.kaggle.com/datasets/mehrdat/coral-reef-global-bleaching]|
| **Classes** | # (name 1, name2) |
| **Format** | CSV file containing metrics like temperature, sea level, and bleaching percentages|
| **Processing** | Structured dataset used to train a separate dense network to predict bleaching risk based on factors such as location, time, and environment.|



- Where your API can be accessed
- ...

# API
Document main API endpoints here

# Setup instructions
These instructions are for users who wish to clone the repository and run the training scripts locally.

# Prerequisites
You need Python 3.8+ installed.

|**Clone the Repository**|:
git clone https://github.com/YourUsername/ReefSight-Project.git
cd ReefSight-Project

|**Create and Activate Virtual Environment**|:
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows


|**Install Dependencies**|:

pip install -r requirements.txt

|**Data Preparation**|:
Download Data: Download the zip files from the Kaggle links.

**Place Data**: Create the required data structure within the project root:

/ReefSight-Project
|-- /raw_data
    |-- /Bleached_and_Unbleached_Corals_Classification/
        |-- archive.zip  <-- PLACE THE DOWNLOADED ZIP FILE HERE
Run Data Setup: The main script will automatically unzip and split the data into train, val, and test folders.

# Usage
The core functionality of this package is the training and comparison of the deep learning models.

### Model Training Pipeline

The script runs three distinct training experiments:

1.  **Baseline CNN:** A custom-built convolutional network.
2.  **VGG16 Frozen:** Transfer learning using a pre-trained VGG16 base (frozen) for feature extraction.
3.  **VGG16 Augmented:** The VGG16 model combined with on-the-fly Data Augmentation (random rotation, zoom, and flip) for enhanced robustness.

To run the full training pipeline, execute the main script:
bash
python main.py
