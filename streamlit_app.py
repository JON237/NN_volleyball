import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Datensatz nur einmal laden und zwischenspeichern
@st.cache_data
def load_data(pfad: str):
    """Liest das CSV mit Teamnamen ein."""
    return pd.read_csv(pfad)

@st.cache_resource
def train_model(df: pd.DataFrame):
    """Skaliert die Daten, baut das Modell auf und trainiert es."""

    # Diese numerischen Merkmale dienen als Eingaben
    numeric_cols = [
        "home_rank",
        "away_rank",
        "home_wins_last_10",
        "away_wins_last_10",
        "home_points_last_match",
        "away_points_last_match",
    ]

    # Teamnamen werden in Integer-IDs umgewandelt
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

    # Aufbau des funktionalen Modells
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
    # Optimierer, Verlustfunktion und Metrik festlegen
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Early-Stopping verhindert Überanpassung
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
    st.title("Vorhersage von Volleyballspielen")

    # Daten laden und Modell beim Start trainieren
    df = load_data("volleyball_matches_with_teams.csv")
    model, scaler, team_to_idx = train_model(df)
    teams = list(team_to_idx.keys())

    st.header("Team-Statistiken")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Heimteam")
        home_team = st.selectbox("Team", teams)
        home_rank = st.number_input("Rang", value=10, min_value=1)
        home_wins_last_10 = st.number_input("Siege der letzten 10", value=5, min_value=0)
        home_points_last_match = st.number_input("Punkte letztes Spiel", value=75, min_value=0)
    with col2:
        st.subheader("Auswärtsteam")
        away_team = st.selectbox("Team ", teams, index=1)
        away_rank = st.number_input("Rang ", value=12, min_value=1)
        away_wins_last_10 = st.number_input("Siege der letzten 10 ", value=4, min_value=0)
        away_points_last_match = st.number_input("Punkte letztes Spiel ", value=70, min_value=0)

    if st.button("Gewinner vorhersagen"):
        # Eingaben wie beim Training vorbereiten
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
        # Wahrscheinlichkeit je nach Team interpretieren
        if prob > 0.5:
            winner = "Heimteam"
        else:
            winner = "Auswärtsteam"
            prob = 1.0 - prob
        st.write(f"**{winner} gewinnt wahrscheinlich mit {prob:.2f} Wahrscheinlichkeit**")


if __name__ == "__main__":
    main()
