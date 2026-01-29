<div align="center">
  <img src="assets/banner.gif" alt="ReefSight Scanning Banner" width="100%">
</div>

<br />

# ReefSight

**🌊 Description**

**ReefSight** is a multimodal deep learning application designed to predict coral bleaching risk by combining both high-resolution reef imagery and environmental (tabular) data. By leveraging computer vision, transfer learning, and structured environmental predictors, ReefSight offers a scalable, non-invasive approach to coral reef health monitoring — a key tool for timely conservation efforts in the face of climate change.

**🌎 Project Motivation**

Coral reefs support up to 25% of all marine species and are vital for biodiversity, fisheries, and coastal protection. However, rising ocean temperatures and other stressors are causing more frequent and severe coral bleaching events. Traditional monitoring methods can be resource-intensive and limited in scale. ReefSight seeks to automate coral health predictions using AI, enabling faster insights for researchers and conservationists

---

## 🧠 What ReefSight Does

ReefSight supports **three prediction modes**:

1. **Image-only prediction**
   Uses a trained deep-learning image model to classify coral health from uploaded reef photos.

2. **Tabular-only prediction**
   Predicts bleaching risk using environmental variables such as temperature, depth, turbidity, wind speed, and cyclone frequency.

3. **Multi-modal fusion (frontend-level)**
   Allows users to combine image input and environmental context within the same analysis workflow (models remain independently trained).

---

## 📚 Datasets Used

ReefSight relies on **publicly available datasets** for both image-based and environmental (tabular) modeling. These datasets were selected for their relevance to coral bleaching research and their suitability for machine-learning workflows.

### 🖼️ Coral Reef Image Dataset

* **Name**: *Bleached Corals Detection*
* **Source**: Kaggle
* **URL**: [https://www.kaggle.com/datasets/sonainjamil/bleached-corals-detection](https://www.kaggle.com/datasets/sonainjamil/bleached-corals-detection)

**Description**:
This dataset contains labeled underwater images of coral reefs categorized as *bleached* and *non-bleached*. It is used to train and evaluate convolutional neural networks for visual coral health classification.

**Disclaimer**:
The original Kaggle dataset was curated by removing non-coral imagery and augmented with Internet-sourced images of healthy and bleached corals.

**Usage in ReefSight**:

* Image preprocessing and augmentation
* CNN baseline model training
* VGG16 transfer learning experiments

---

### 📊 Global Coral Bleaching (Tabular) Dataset

* **Name**: *Coral Reef Global Bleaching Dataset*
* **Source**: Kaggle
* **URL**: [https://www.kaggle.com/datasets/mehrdat/coral-reef-global-bleaching](https://www.kaggle.com/datasets/mehrdat/coral-reef-global-bleaching)

**Description**:
A global dataset documenting coral bleaching observations alongside environmental and geographic variables such as temperature, depth, turbidity, and cyclone exposure.

**Usage in ReefSight**:

* Exploratory data analysis (EDA)
* Feature selection and validation
* Training of the tabular bleaching prediction model

---

### 🌐 External Environmental Data Sources

In addition to static datasets, ReefSight derives dynamic environmental features using **external climatological data sources**, primarily inspired by:

* **NOAA Coral Reef Watch** indicators
* Public oceanographic and meteorological datasets

These sources support automated feature enrichment based on geospatial coordinates and observation dates.

---

## 🏗️ Architecture Overview

```
Streamlit App (Frontend)
        ↓ HTTP requests
FastAPI Backend (Inference)
        ↓
Trained Image Model + Trained Tabular Model
```

* **Frontend**: `Streamlit` (`app.py`)
* **Backend API**: `FastAPI` (`api/fast.py`)
* **ML Logic**: `project_logic/`
* **Model Training & EDA**: `notebooks/`

---

## 📁 Repository Structure

```
ReefSight-Project/
├── api/
│   └── fast.py              # FastAPI inference service
├── project_logic/
│   ├── predict.py           # Image & tabular prediction logic
│   ├── preprocessing.py     # Pydantic input schemas
│   ├── tabular_fetch.py     # NOAA & environmental data enrichment
│   ├── tabular_preproc.py   # Feature engineering
├── notebooks/
│   ├── IMAGE_MODEL_*.ipynb  # CNN & VGG16 model experiments
│   ├── coral_bleaching_tabular_pipe.ipynb
│   └── *_EDA.ipynb
├── app.py                   # Streamlit frontend
├── Dockerfile               # API containerization
├── Makefile                 # Dev & deployment helpers
├── requirements.txt
└── setup.py
```

---

## 🖼️ Image Prediction Pipeline

* Accepts uploaded reef images (`.jpg`, `.png`, `.jpeg`)
* Images are processed as raw bytes
* Predictions are performed using a **pre‑trained CNN‑based image model**
* Implemented in:

  * `project_logic.predict.predict_image()`

### API Endpoint

```
POST /predict/image
```

**Input**: Image file
**Output**: Bleaching classification prediction

---

## 📊 Tabular Prediction Pipeline

The tabular model supports **two input modes**:

### 1. Minimal Input

```json
{
  "latitude": -18.2,
  "longitude": 147.7,
  "observation_date": "2023-08-01"
}
```

Environmental features are **automatically fetched and derived** using NOAA‑based data pipelines.

### 2. Full Feature Override

Users may optionally supply a complete environmental feature vector including:

* Distance to shore
* Turbidity
* Cyclone frequency
* Depth
* Sea surface temperature
* Temperature variability
* Wind speed

### API Endpoint

```
POST /predict/tabular
```

Implemented in:

* `project_logic.tabular_fetch.build_X_pred()`
* `project_logic.predict.predict_tabular()`

---

## 🔍 Data Preprocessing (Tabular Pipeline)

A substantial portion of ReefSight focuses on **robust preprocessing and feature engineering** for environmental (tabular) data, ensuring that bleaching predictions are scientifically grounded and reproducible.

### Input Modes

The tabular pipeline supports two levels of input:

1. **Minimal Geospatial Input**
   Users provide latitude, longitude, and observation date. From these, the pipeline automatically derives environmental predictors.

2. **Full Feature Override**
   Advanced users may supply a complete environmental feature vector, bypassing auto‑fetching and derivation.

### Data Enrichment

Environmental variables are programmatically fetched and derived using external climatological sources (e.g. NOAA‑based datasets). The pipeline computes and aggregates:

* Sea surface temperature (mean, variability)
* Wind speed
* Cyclone frequency
* Turbidity
* Depth and distance to shore

This logic is implemented primarily in:

* `project_logic/tabular_fetch.py`

### Feature Engineering & Cleaning

Before inference, tabular data undergoes:

* Type validation via **Pydantic schemas**
* Missing value handling and default imputation
* Scaling and normalization aligned with training distributions
* Feature ordering to match trained model expectations

Implemented in:

* `project_logic/tabular_preproc.py`
* `project_logic/preprocessing.py`

---

## 🧾 Tabular Feature Glossary

The following features are used by the tabular bleaching prediction model. When not explicitly provided by the user, they are **automatically derived** from geospatial coordinates and observation date.

| Feature                   | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `latitude` / `longitude`  | Geographic coordinates of the reef observation           |
| `observation_date`        | Date of observation, used for temporal aggregation       |
| `sea_surface_temperature` | Mean sea surface temperature over the observation window |
| `sst_variability`         | Short-term variability in sea surface temperature        |
| `wind_speed`              | Average surface wind speed near the reef location        |
| `cyclone_frequency`       | Historical frequency of cyclonic activity in the region  |
| `turbidity`               | Proxy for water clarity and suspended particles          |
| `depth`                   | Estimated reef depth at the given coordinates            |
| `distance_to_shore`       | Distance from reef location to nearest shoreline         |

These features were selected based on **established coral bleaching literature** and exploratory data analysis performed during model development.

---

## 📊 Model Performance Summary

### Image Model

* Architecture: CNN baseline and VGG16 (transfer learning, trained offline)
* Task: Binary classification (bleached vs non‑bleached coral imagery)
* Evaluation metrics explored in notebooks:

  * Accuracy
  * Precision / Recall
  * Confusion Matrix

Transfer learning models demonstrated **notably improved generalization** compared to the baseline CNN, particularly on validation data.

### Tabular Model

* Input: Engineered environmental feature vectors
* Task: Bleaching risk classification
* Focus areas:

  * Feature importance analysis
  * Sensitivity to temperature‑based predictors

Results indicate that **temperature‑related variables and variability metrics** are among the strongest predictors of bleaching risk.

> Detailed metrics, plots, and comparisons are documented in the notebooks within the `notebooks/` directory.

---

## 🚀 Running the Backend Locally

This repository contains the **backend inference API and machine-learning logic** for ReefSight.
The interactive frontend application is maintained in a **separate repository**:

👉 [https://github.com/Lucia-Cordero/ReefSight_front](https://github.com/Lucia-Cordero/ReefSight_front)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
uvicorn api.fast:app --reload
```

Once running, the API will be available at:

```
http://localhost:8000
```

### 3. Test the API

You can interact with the backend using:

* FastAPI Swagger UI (`/docs`)
* API clients such as `curl` or Postman
* The ReefSight frontend application (separate repository)

---

## 🐳 Docker Support

The backend API can be containerized using the provided `Dockerfile`.

```bash
docker build -t reefsight-api .
docker run -p 8000:8000 reefsight-api
```

---

## 🧪 Model Development

All model training and experimentation lives in the `notebooks/` directory and includes:

* Image dataset exploration
* CNN baseline model
* VGG16 transfer learning
* Tabular feature engineering and pipelines

These notebooks document the **research and modeling process** behind the deployed models.

---

## 📌 Limitations & Notes

* Image and tabular models are trained **independently**
* "Multi‑modal" fusion occurs at the **application level**, not via a joint neural architecture
* Predictions are intended for **research and educational purposes**, not operational reef management

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

* NOAA Coral Reef Watch
* Open coral bleaching datasets
* Streamlit & FastAPI open‑source communities

---

🌊 *ReefSight aims to make coral health monitoring more accessible, interpretable, and scalable through AI.*
