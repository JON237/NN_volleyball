import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load and train model only once
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("result", axis=1)
    y = df["result"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(X_scaled.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_scaled, y, epochs=50, batch_size=8, verbose=0)
    return model, scaler


def main():
    st.title("Volleyball Match Outcome Predictor")
    df = load_data("volleyball_matches.csv")
    model, scaler = train_model(df)

    st.header("Team Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home Team")
        home_rank = st.number_input("Rank", value=10, min_value=1)
        home_wins_last_10 = st.number_input("Wins in last 10", value=5, min_value=0)
        home_points_last_match = st.number_input("Points last match", value=75, min_value=0)
    with col2:
        st.subheader("Away Team")
        away_rank = st.number_input("Rank ", value=12, min_value=1)
        away_wins_last_10 = st.number_input("Wins in last 10 ", value=4, min_value=0)
        away_points_last_match = st.number_input("Points last match ", value=70, min_value=0)

    if st.button("Predict Winner"):
        features = pd.DataFrame([
            [
                home_rank,
                away_rank,
                home_wins_last_10,
                away_wins_last_10,
                home_points_last_match,
                away_points_last_match,
            ]
        ], columns=[
            "home_rank",
            "away_rank",
            "home_wins_last_10",
            "away_wins_last_10",
            "home_points_last_match",
            "away_points_last_match",
        ])
        scaled = scaler.transform(features)
        prob = float(model.predict(scaled)[0][0])
        if prob > 0.5:
            winner = "Home team"
        else:
            winner = "Away team"
            prob = 1.0 - prob
        st.write(f"**{winner} will likely win with probability {prob:.2f}**")


if __name__ == "__main__":
    main()
