import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the CSV data
def load_data(filepath):
    return pd.read_csv(filepath)

# Filter the data
def filter_data(data):
    historical_matches = data[data['match_status'] != 'not_started']
    upcoming_matches = data[data['match_status'] == 'not_started']
    return historical_matches, upcoming_matches

# Create a function to determine match outcome
def match_outcome(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] < row['away_score']:
        return 'away_win'
    else:
        return 'draw'

# Apply the function to historical matches
def add_outcome_column(historical_matches):
    historical_matches['outcome'] = historical_matches.apply(match_outcome, axis=1)
    return historical_matches

# Calculate win rates for home and away teams
def team_stats(matches, team, location):
    if location == 'home':
        wins = matches[(matches['home_name'] == team) & (matches['outcome'] == 'home_win')].shape[0]
        total = matches[matches['home_name'] == team].shape[0]
    else:
        wins = matches[(matches['away_name'] == team) & (matches['outcome'] == 'away_win')].shape[0]
        total = matches[matches['away_name'] == team].shape[0]
    return wins / total if total > 0 else 0

# Add features to historical matches
def add_features(historical_matches):
    historical_matches['home_win_rate'] = historical_matches['home_name'].apply(lambda x: team_stats(historical_matches, x, 'home'))
    historical_matches['away_win_rate'] = historical_matches['away_name'].apply(lambda x: team_stats(historical_matches, x, 'away'))
    return historical_matches

# Prepare training data
def prepare_training_data(historical_matches):
    X = historical_matches[['home_win_rate', 'away_win_rate']]
    y = historical_matches['outcome'].map({'home_win': 0, 'away_win': 1, 'draw': 2})
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Add features to upcoming matches
def add_upcoming_features(upcoming_matches, historical_matches):
    upcoming_matches['home_win_rate'] = upcoming_matches['home_name'].apply(lambda x: team_stats(historical_matches, x, 'home'))
    upcoming_matches['away_win_rate'] = upcoming_matches['away_name'].apply(lambda x: team_stats(historical_matches, x, 'away'))
    return upcoming_matches[['home_win_rate', 'away_win_rate']]

# Predict outcomes for upcoming matches
def predict_outcomes(model, X_upcoming):
    predictions = model.predict(X_upcoming)
    outcome_map = {0: 'home_win', 1: 'away_win', 2: 'draw'}
    return [outcome_map[pred] for pred in predictions]

def main():
    # Load data
    filepath = '/Users/harpallpurewal/Desktop/Euro\ Predictor\ 2024/EM\ 2024\ pairings\ \(1\).csv'
    data = load_data(filepath)

    # Filter data
    historical_matches, upcoming_matches = filter_data(data)

    # Debugging: Print the shape of the datasets
    print(f'Historical matches shape: {historical_matches.shape}')
    print(f'Upcoming matches shape: {upcoming_matches.shape}')

    # Add outcome column to historical matches
    historical_matches = add_outcome_column(historical_matches)

    # Add features to historical matches
    historical_matches = add_features(historical_matches)

    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(historical_matches)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Add features to upcoming matches
    X_upcoming = add_upcoming_features(upcoming_matches, historical_matches)

    # Predict outcomes for upcoming matches
    upcoming_matches['predicted_outcome'] = predict_outcomes(model, X_upcoming)

    # Debugging: Print predictions for all upcoming matches
    print(upcoming_matches[['home_name', 'away_name', 'predicted_outcome']])

    # Print the predicted outcome for England vs Spain
    match_england_spain = upcoming_matches[
        (upcoming_matches['home_name'] == 'England') & (upcoming_matches['away_name'] == 'Spain')
    ]
    if not match_england_spain.empty:
        print("Prediction for England vs Spain:", match_england_spain['predicted_outcome'].values[0])
    else:
        print("Match England vs Spain not found in upcoming matches.")

if __name__ == "__main__":
    main()
