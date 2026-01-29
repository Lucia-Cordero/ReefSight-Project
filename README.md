<div align="center">
  <img src="assets/banner.gif" alt="ReefSight Scanning Banner" width="100%">
</div>

<br />

# ReefSight

**🌊 Description**

**ReefSight** is a multimodal deep learning application designed to predict coral bleaching risk by combining both high-resolution reef imagery and environmental (tabular) data. By leveraging computer vision, transfer learning, and structured environmental predictors, ReefSight offers a scalable, non-invasive approach to coral reef health monitoring — a key tool for timely conservation efforts in the face of climate change.

**🚀 Key Features**

|**Image-based Classification**|: Automated download, formatting, and splitting of raw image data; implementation of Baseline CNN and VGG16 Transfer Learning.

|**Tabular Data-based Classification**|: Use of structured data from the Global Coral Reef Monitoring Network (GCRMN) dataset to identify environmental risk factors.

|**Performance Comparison**|: Direct comparison between image-only, tabular-only, and the final Multi-Modal Fusion Model to determine the optimal solution. develops deep learning models to automatically classify coral images as either 'Bleached' or 'Unbleached'. By leveraging computer vision and transfer learning, we aim to provide a reliable, scalable, and non-invasive tool for monitoring coral reef health, accelerating conservation efforts.

|**Multi-Modal Data Integration**|: Combines coral imagery and structured environmental data to improve prediction accuracy.


## 📊 Data used

### Image Classifier
|**Source** | Kaggle Dataset: Bleached Corals Detection |

| **URL** | [https://www.kaggle.com/datasets/sonainjamil/bleached-corals-detection]|

| **Classes** | 2 (Bleached, Unbleached) |

| **Format** | JPG images |

| **Preprocessing** | Images are resized to **224x224** pixels and rescaled to $[0, 1]$ before model input. |

| **Splitting Ratio**| Data is automatically split into **80% Training, 10% Validation, 10% Testing** sets. |


### Tabular Classifier

|**Source** | Dataset: Bleaching and environmental data for global coral reef sites from 1980-2020|

| **URL** | [https://www.bco-dmo.org/dataset/773466]|

| **Classes** | # (name 1, name2) |

| **Format** | CSV file containing metrics like temperature, sea level, and bleaching percentages|

| **Processing** | Structured dataset used to train a separate dense network to predict bleaching risk based on factors such as location, time, and environment.|




## 📦 Setup instructions
These instructions are for users who wish to clone the repository and run the training scripts locally.

### Prerequisites
You need Python 3.10.6 installed.

|**Clone the Repository**|:
git clone https://github.com/YourUsername/ReefSight-Project.git
cd ReefSight-Project
