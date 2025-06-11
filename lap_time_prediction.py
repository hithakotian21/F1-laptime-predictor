import streamlit as st
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Enable caching to avoid re-downloading data
fastf1.Cache.enable_cache('cache')

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="F1 Lap Time Predictor", layout="centered")
st.title("🏎️ F1 Lap Time Predictor")
st.markdown("Enter a **driver** and a **circuit**, and we’ll predict lap times using past qualifying data.")

# Inputs
# List of F1 drivers (2023 abbreviations)
driver_list = [
    'VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA',
    'OCO', 'GAS', 'ALO', 'STR', 'MAG', 'HUL', 'TSU', 'RIC',
    'SAR', 'ALB', 'BOT', 'ZHO', 'DEV', 'LAW'
]

# List of 2023 circuit names recognized by FastF1
circuit_list = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami',
    'Emilia Romagna', 'Monaco', 'Spain', 'Canada', 'Austria',
    'Great Britain', 'Hungary', 'Belgium', 'Netherlands', 'Italy',
    'Singapore', 'Japan', 'Qatar', 'United States', 'Mexico',
    'Brazil', 'Las Vegas', 'Abu Dhabi'
]

# Streamlit dropdowns
driver_input = st.selectbox("Choose a Driver", driver_list, index=0)
circuit_input = st.selectbox("Choose a Circuit", circuit_list, index=14)  # Default to 'Italy' (Monza)

# Input: Prediction Year
target_year = st.selectbox("Select Year to Predict", list(range(2019, 2026)), index=6)



import random

# Randomized defaults
compound_map = {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1}
compound_label = random.choice(list(compound_map.keys()))
compound_value = compound_map[compound_label]

tyre_life = random.randint(1, 20)

track_status_map = {
    'Green Flag': 1,
    'Yellow Flag': 2,
    'Red Flag': 3,
    'Safety Car': 4,
}
track_status_label = random.choice(list(track_status_map.keys()))
track_status_value = track_status_map[track_status_label]



# Load data on button click
if st.button("Predict Lap Times"):
    with st.spinner("Fetching and processing data..."):
        years = [2019, 2020, 2021, 2022, 2023]
        all_laps = []

        for year in years:
            try:
                session = fastf1.get_session(year, circuit_input, 'Qualifying')
                session.load()
                laps = session.laps.pick_quicklaps()
                driver_laps = laps[laps['Driver'] == driver_input.upper()]
                if not driver_laps.empty:
                    df = driver_laps[['Driver', 'LapTime', 'Compound', 'TrackStatus',
                                      'Sector1Time', 'Sector2Time', 'Sector3Time', 'TyreLife']].copy()
                    df['Year'] = year
                    all_laps.append(df)
            except Exception as e:
                st.warning(f"⚠️ Could not load data for {circuit_input} {year}: {e}")

        # Check if data exists
        if not all_laps:
            st.error("No data found for that driver and circuit over the past 5 years.")
        else:
            data = pd.concat(all_laps)
            data.dropna(inplace=True)

            # Convert time columns to seconds
            data['LapTime'] = data['LapTime'].dt.total_seconds()
        

            # Encode compound
            data['Compound'] = LabelEncoder().fit_transform(data['Compound'])

            # Features and target
            features = ['Year','Compound', 'TrackStatus', 'TyreLife']
            X = data[features]
            y = data['LapTime']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.markdown("### 🛠️ Input Parameters (Randomized)")
            st.info(f"- **Tyre Compound:** {compound_label}\n"
                    f"- **Tyre Life:** {tyre_life} laps\n"
                    f"- **Track Status:** {track_status_label}")


            st.success("✅ Prediction Complete!")
            st.markdown(f"**Mean Absolute Error (MAE):** `{mae:.2f} seconds`")
            st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f} seconds`")

            # Prepare prediction input (same structure as X)
            predict_input = pd.DataFrame([{
            "Year": target_year,
            "Compound": compound_value,        # Assume SOFT
            "TyreLife": tyre_life,        # Fresh tyre
            "TrackStatus": track_status_value,
            }])


            # Predict lap time for the selected year
            predicted_time = model.predict(predict_input)[0]

            # Output results
            st.markdown("### 🏁 Predicted Lap Time")
            st.success(f"Estimated Lap Time for **{driver_input}** at **{circuit_input}** in **{target_year}**: **{predicted_time:.2f} seconds**")

            

            # Plot
            st.subheader("📊 Actual vs Predicted Lap Times")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax, label='Predicted Points')

            # Add y = x line for perfect prediction reference
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y = x)')

            ax.set_xlabel("Actual Lap Time (s)")
            ax.set_ylabel("Predicted Lap Time (s)")
            ax.set_title(f"{driver_input.upper()} @ {circuit_input} (2019–2023)")
            ax.legend()
            st.pyplot(fig)
