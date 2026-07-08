# -lungsheba
AI-powered Chest X-ray Abnormality Detector for Bangladesh
# 🫁 LungSheba | লাং সেবা
### AI-powered Chest X-ray Abnormality Detector

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/TheRealShaswata/lungsheba)

---

## 📌 What it does
Upload a chest X-ray image → AI detects abnormalities → shows urgency level → 
explains in simple Bangla → tells you which doctor/hospital to visit.

## 🎯 Conditions Detected
| Condition | Urgency |
|---|---|
| Pneumonia | 🔴 HIGH |
| Edema | 🔴 HIGH |
| Cardiomegaly | 🔴 HIGH |
| Consolidation | 🟡 MEDIUM |
| No Finding | 🟢 NORMAL |

## 🛠️ Tech Stack
- **Model:** Vision Transformer (ViT) — fine-tuned on CheXpert dataset
- **Framework:** Streamlit
- **Hosting:** Hugging Face Spaces (free)
- **Language:** Python

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## ⚠️ Disclaimer
This tool is for awareness only. It is NOT a medical diagnosis.
Always consult a qualified doctor.

## 📊 Dataset
- CheXpert — Stanford University chest X-ray dataset
- NIH ChestX-ray14 — 112,120 frontal X-ray images, 14 disease labels

## 👨‍💻 Team
**Bytestorm** — SciBlitz AI Challenge 2026
