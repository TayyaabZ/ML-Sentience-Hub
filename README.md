# ⚡ The ML Sentience Hub

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-4.1.1-E25A1C?style=for-the-badge&logo=apachespark)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Java](https://img.shields.io/badge/Java-21_LTS-007396?style=for-the-badge&logo=openjdk)

**A Dual-Pipeline Big Data Analytics System for Real-Time Threat Detection and Grid Failure Prediction.**

This repository contains the source code for a final-year Big Data Analytics project. It demonstrates a complete, end-to-end Apache Spark machine learning pipeline, including synthetic data generation, distributed model training, leak-free data transformations, and a live interactive inference dashboard.

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Execution Instructions](#-execution-instructions-reproducibility)
- [Authors](#-authors)

---

## ✨ Features

The system addresses two distinct real-world problems through independent PySpark processing pipelines:

### 1. Module Alpha: Aegis-Vanguard (E-Commerce Security)
* **Problem:** Detecting automated botnet scalpers during high-demand product releases.
* **Algorithm:** Random Forest Classifier.
* **Performance:** 100% Accuracy / 100% F1 Score.

### 2. Module Beta: Nexus-Grid (Smart City Infrastructure)
* **Problem:** Predicting transformer overloads in the Islamabad/Rawalpindi electricity grid using IoT telemetry.
* **Algorithm:** Gradient Boosted Trees (GBT).
* **Performance:** 100% Accuracy / 100% F1 Score.

### 3. The Visualization Hub
* A unified, dark-themed Streamlit dashboard providing real-time inference and historical analytics.
* Utilizes serialized PySpark `PipelineModel` objects to serve instant predictions without retraining overhead.

---

## 🏗️ System Architecture

* **Data Strategy:** Models are trained on dynamically generated, 150,000-row synthetic datasets. Data includes intentional missing values and corrupted records to validate cleaning pipelines.
* **Leak-Free Pipelines:** Train/test splits are strictly enforced *before* any data transformation (Null Imputation, StringIndexing, VectorAssembly) to ensure zero data leakage.
* **Deployment Mode:** PySpark running in local cluster mode (`local[*]`).

---

## 📂 Project Structure

    The ML Sentience Hub/
    ├── data_forge.py                  # Synthetic dataset generator
    ├── dashboard.py                   # Unified Streamlit UI
    ├── requirements.txt               # Python dependencies
    ├── aegis_vanguard/
    │   └── bot_detector.py            # Module Alpha ML pipeline
    └── nexus_grid/
        └── grid_predictor.py          # Module Beta ML pipeline

*(Note: `data/` and `models/` directories are intentionally excluded from version control due to file size constraints. The pipeline is designed to be fully reproducible from source.)*

---

## 🚀 Execution Instructions (Reproducibility)

Follow these steps to build the datasets, train the models, and launch the UI from scratch.

### 1. Prerequisites
* **OS:** Linux (Fedora recommended)
* **Python:** 3.14
* **Java:** OpenJDK 21 LTS

### 2. Setup Environment
Clone the repository and install dependencies:

    git clone <your-github-repo-url>
    cd ML-Sentience-Hub
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt


### 3. Generate the Datasets
Run the data forge to generate `aegis_raw_logs.csv` and `nexus_raw_telemetry.csv`.

    python data_forge.py


### 4. Train & Serialize the Models
Run the distributed training pipelines. These scripts clean the data, train the classifiers, and save the models to a local `models/` directory.

    python aegis_vanguard/bot_detector.py
    python nexus_grid/grid_predictor.py


### 5. Launch the Sentience Hub
Start the Streamlit web application for live inference:

    streamlit run dashboard.py


---

## 👥 Authors

* **Muhammad Tayyaab Zahoor**
* **Anood Tayyeba Imtiaz**

**Course:** Big Data Analytics (Spring 2026)
**Instructor:** Ms. Ameena Saeed
