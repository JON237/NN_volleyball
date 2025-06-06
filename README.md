# NN_volleyball
a neural network for predicting  results of volleyball matches

This repository contains a minimal example of training a neural network to predict the outcome of volleyball matches.

## Dataset

The training data is expected as a CSV file where each row represents one past match. An example dataset `volleyball_matches_with_teams.csv` is included with team names from the German Bundesliga:

```
ASV Dachau
Baden Volleys SSC Karlsruhe
BERLIN RECYCLING Volleys
Energiequelle Netzhoppers KW
FT 1844 Freiburg
HELIOS GRIZZLYS Giesen
SVG Lüneburg
SWD powervolleys DÜREN
TSV Haching München
VC Bitterfeld-Wolfen
VCO Berlin
VfB Friedrichshafen
WWK Volleys Herrsching
```

The CSV columns used are:

- `home_team` – name of the home team (14 different teams)
- `away_team` – name of the away team
- `home_rank` – rank of the home team
- `away_rank` – rank of the away team
- `home_wins_last_10` – wins by the home team in the last 10 matches
- `away_wins_last_10` – wins by the away team in the last 10 matches
- `home_points_last_match` – points scored by the home team in the previous match
- `away_points_last_match` – points scored by the away team in the previous match
- `result` – 1 if the home team won, 0 otherwise

## Training

Install the dependencies and run `volleyball_nn.py`:

```bash
pip install tensorflow scikit-learn pandas streamlit
python volleyball_nn.py --data volleyball_matches_with_teams.csv
```

The script splits the data into training and test sets, encodes the team names through an embedding layer and scales the numeric features. It then trains a TensorFlow network with layers 64-32-16 and prints the test accuracy.

## Interactive Demo

To experiment with your own inputs, launch the Streamlit application:

```bash
streamlit run streamlit_app.py --server.headless true
```

The web interface lets you enter team statistics and returns which team the model expects to win along with the predicted probability. statistics and returns which team the model expects to win along with the predicted probability.
