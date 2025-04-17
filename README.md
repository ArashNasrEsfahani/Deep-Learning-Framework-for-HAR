 
# A Deep Learning Framework for Human Activity Recognition*

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

---

## 📑 Table of Contents
- [🌟 Overview](#-overview)  
- [✨ Features](#-features)  
- [📂 Datasets](#-datasets)  
- [🏗️ Model Architectures](#️-model-architectures)  
- [📊 Results](#-results)  

---

## 🌟 Overview
This repo contatins notebook for HAR that are modular, end‑to‑end pipeline for human activity recognition (HAR) using wearable sensor data. They implement and benchmark four architectures to extract both spatial and temporal features, plus attention mechanisms for refined performance.

---

## ✨ Features
- **Flexible Preprocessing**: normalization, segmentation & class balancing  
- **Plug‑and‑Play Models**: easily switch between LSTM, attention, CNN–LSTM, and hybrid designs  
- **Training Utilities**: configurable epochs, early stopping & checkpointing  
- **Evaluation Tools**: accuracy metrics, confusion matrices & classification reports  
- **Reproducible Notebooks**: ready for Google Colab experiments  

---

## 📂 Datasets
1. **UCI‑HAR**  
   - 30 subjects, 6 activities, 561 features  
   - Pre‑segmented into train/test  
2. **MHEALTH**  
   - 10 subjects, originally 12 → filtered to 4 activities  
   - Scripts handle 70/15/15 split & balancing  

Place your CSVs under `data/uci_har/` and `data/mhealth/` before running.

---

## 🏗️ Model Architectures
- **LSTM**  
  Two-layer LSTM (128 → 64 units) with dropout  
- **LSTM + Self‑Attention**  
  LSTM → attention → LSTM → dropout  
- **CNN + LSTM + Self‑Attention**  
  Two 1D conv layers → LSTM → attention  
- **State‑of‑the‑Art Hybrid**  
  Residual 1D conv blocks → Bi‑LSTM → multi‑head attention  

Full definitions live in the `models/` folder.

---

## 📊 Results

| Dataset   | Model                         | Accuracy (%) |
|-----------|-------------------------------|-------------:|
| **UCI‑HAR** | LSTM                          |        93.76 |
|           | LSTM + Attention              |        94.40 |
|           | CNN + LSTM + Attention        |        94.91 |
|           | **State‑of‑the‑Art Hybrid**   |   **95.24**  |
| **MHEALTH** | LSTM                          |        97.98 |
|           | LSTM + Attention              |        98.24 |
|           | CNN + LSTM + Attention        |        98.01 |
|           | **State‑of‑the‑Art Hybrid**   |   **97.93**  |


