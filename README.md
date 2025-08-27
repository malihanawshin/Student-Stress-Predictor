# Student-Stress-Level-Predictor

This project predicts **student stress levels** using survey data covering psychological, physiological, environmental, academic, and social factors. The goal is to demonstrate a **machine learning pipeline** with a simple and interactive **Streamlit web app**.

## Dataset

The dataset is derived from Kaggle (citation below), which was based on a nationwide student survey and includes **20 features** grouped under five categories:

* **Psychological**: anxiety\_level, self\_esteem, mental\_health\_history, depression
* **Physiological**: headache, blood\_pressure, sleep\_quality, breathing\_problem
* **Environmental**: noise\_level, living\_conditions, safety, basic\_needs
* **Academic**: academic\_performance, study\_load, teacher\_student\_relationship, future\_career\_concerns
* **Social**: social\_support, peer\_pressure, extracurricular\_activities, bullying

The target label is **`stress_level`**.

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/stress-prediction.git
cd stress-prediction

# create virtual environment (recommended)
conda create -n stressenv python=3.11
conda activate stressenv

# install dependencies
pip install -r requirements.txt
```

## How to Run

### 1. Train the Model

```bash
python train.py
```

This will train a **Decision Tree model** and save it in the `models/` folder.

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will launch locally, allowing you to input student factors and predict stress levels.

## Deployment

This app can be deployed on **Streamlit Cloud** or **AWS EC2** for live demo access.

## Demo Features

* Input 20 stress-related factors
* Predict student stress level instantly
* Interactive Streamlit UI
* Lightweight ML model

## Citation 

Ovi, M. S., Hossain, J., Rahi, M. R., & Akter, F. (2025). Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring. ArXiv. https://arxiv.org/abs/2508.01105
