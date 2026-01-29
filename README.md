<div align="center">
  <img src="assets/banner.gif" alt="ReefSight Scanning Banner" width="100%">
</div>

<br />

# ReefSight

**🌊 Description**

**ReefSight** is a multimodal deep learning application designed to predict coral bleaching risk by combining both high-resolution reef imagery and environmental (tabular) data. By leveraging computer vision, transfer learning, and structured environmental predictors, ReefSight offers a scalable, non-invasive approach to coral reef health monitoring — a key tool for timely conservation efforts in the face of climate change.

**🚀 Key Features**

**Image-based Classification**: Automated download, formatting, and splitting of raw image data; implementation of Baseline CNN and VGG16 Transfer Learning.

**Tabular Data-based Classification**: Use of structured data from the Global Coral Reef Monitoring Network (GCRMN) dataset to identify environmental risk factors.

**Multi-Modal Data Integration**: Combines coral imagery and structured environmental data to improve prediction accuracy.


## 📊 Data used

### Image Classifier

| **Source** | Manually curated dataset derived from Kaggle Dataset: "Bleached Corals Detection"

| **URL** | [https://www.kaggle.com/datasets/sonainjamil/bleached-corals-detection]

| **Classes** | 2 (Bleached, Unbleached)

| **Format** | JPG images




### Tabular Classifier

| **Source** | Dataset: "Bleaching and environmental data for global coral reef sites from 1980-2020"

| **URL** | [https://www.bco-dmo.org/dataset/773466]

| **Classes** | NaN (percentage bleaching --> continuous data)

| **Format** | CSV file containing 62 features and 41361 entries. Features include metrics such as sea_surface_temperature, date_year, turbidity, distance_to_shore, etc.






## 📦 Setup instructions
These instructions are for users who wish to clone the repository and run the training scripts locally.

### Prerequisites
You need Python 3.10.6 installed.

|**Clone the Repository**|:
git clone https://github.com/YourUsername/ReefSight-Project.git
cd ReefSight-Project
