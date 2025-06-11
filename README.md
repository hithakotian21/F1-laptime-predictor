# 🏎️ F1 Lap Time Predictor

This project uses **machine learning** to predict Formula 1 lap times based on historical qualifying session data. Built with **Streamlit**, **FastF1**, and **Random Forest Regressor**, the app takes a selected driver and circuit as input, randomly chooses compound, tyre life, and track status conditions, and estimates lap time performance.

## 🔧 Technologies Used
- Python 🐍
- Streamlit 📈
- FastF1 🏁
- scikit-learn 🔬
- pandas, NumPy, seaborn, matplotlib

## 📦 Features
- Interactive UI with driver, circuit, and year input.
- Randomized input simulation for tyre compound, life, and track status.
- Predicts lap time using Random Forest trained on past qualifying sessions.
- Visualizes actual vs predicted lap time distribution.

## 🚀 How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run it
   streamlit run lap_time_prediction.py
