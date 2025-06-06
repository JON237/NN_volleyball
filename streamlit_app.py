import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the dataset only once and cache it for faster reloads
@st.cache_data
def load_data(path: str):
    """Read the volleyball match CSV with team names."""
    return pd.read_csv(path)

@st.cache_resource
def train_model(df: pd.DataFrame):
    """Scale the data, build the model with embeddings and train it."""

    numeric_cols = [
        "home_rank",
        "away_rank",
        "home_wins_last_10",
        "away_wins_last_10",
        "home_points_last_match",
        "away_points_last_match",
    ]

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    team_to_idx = {t: i for i, t in enumerate(sorted(teams))}
    df["home_idx"] = df["home_team"].map(team_to_idx)
    df["away_idx"] = df["away_team"].map(team_to_idx)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_cols])
    inputs = [
        df["home_idx"].values,
        df["away_idx"].values,
        X_num,
    ]

    # Build functional model
    home_in = tf.keras.layers.Input(shape=(1,), name="home_team")
    away_in = tf.keras.layers.Input(shape=(1,), name="away_team")
    stats_in = tf.keras.layers.Input(shape=(len(numeric_cols),), name="stats")
    e_home = tf.keras.layers.Embedding(len(team_to_idx), 4)(home_in)
    e_away = tf.keras.layers.Embedding(len(team_to_idx), 4)(away_in)
    x = tf.keras.layers.Concatenate()([
        tf.keras.layers.Flatten()(e_home),
        tf.keras.layers.Flatten()(e_away),
        stats_in,
    ])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[home_in, away_in, stats_in], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Early stopping to avoid overfitting on the small dataset
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        inputs,
        df["result"],
        validation_split=0.2,
        epochs=50,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
    )
    return model, scaler, team_to_idx


def main():
    st.title("Volleyball Match Outcome Predictor")

    # Load data and train the model when the app starts
    df = load_data("volleyball_matches_with_teams.csv")
    model, scaler, team_to_idx = train_model(df)
    teams = list(team_to_idx.keys())

    st.header("Team Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home Team")
        home_team = st.selectbox("Team", teams)
        home_rank = st.number_input("Rank", value=10, min_value=1)
        home_wins_last_10 = st.number_input("Wins in last 10", value=5, min_value=0)
        home_points_last_match = st.number_input("Points last match", value=75, min_value=0)
    with col2:
        st.subheader("Away Team")
        away_team = st.selectbox("Team ", teams, index=1)
        away_rank = st.number_input("Rank ", value=12, min_value=1)
        away_wins_last_10 = st.number_input("Wins in last 10 ", value=4, min_value=0)
        away_points_last_match = st.number_input("Points last match ", value=70, min_value=0)

    if st.button("Predict Winner"):
        # Prepare inputs in the same format used for training
        numeric = pd.DataFrame([
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
        scaled_num = scaler.transform(numeric)
        inputs = [
            [team_to_idx[home_team]],
            [team_to_idx[away_team]],
            scaled_num,
        ]
        prob = float(model.predict(inputs)[0][0])
        # Interpret probability as winning chance of either team
        if prob > 0.5:
            winner = "Home team"
        else:
            winner = "Away team"
            prob = 1.0 - prob
        st.write(f"**{winner} will likely win with probability {prob:.2f}**")


if __name__ == "__main__":
    main()
