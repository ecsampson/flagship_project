from anomaly_data import get_tmax_data
from anomaly_features import build_feature_table
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURE_COLUMNS = ['day_of_year', 'year', 'tmax_lag1']
TARGET_COLUMN = 'tmax'

def split_train_test(df, cutoff_year):
    """Drop rows with no target, then split chronologically on cutoff_year

Args:
    df: feature table from build_feature_table in anomaly_features.py() - needs 'tmax' and 'year' columns.
    cutoff_year: first year to include in the test set. Rows with
        year < cutoff_year go to train; year >= cutoff_year go to test.
        
    
        
Returns:
    (train_df, test_df) tuple.
    """

# Drop rows where we have no ground-truth tmax to train/evaluate on
# These are the gap-days the reindex inserted - nothing to learn or train against
    df = df.dropna(subset=['tmax'])


# Splitting into train or test based off the cutoff year
    train_df = df[df['year'] < cutoff_year]
    test_df = df[df['year'] >= cutoff_year]


    return train_df, test_df


def fit_model(train_df):
    """Fit a GBM regressor predicting the temperature for a given
        day of the year based on previous day's temperatureFit a GBM regressor predicting tmax from seasonal (day_of_year),
long-term trend (year), and short-term persistence (tmax_lag1) features.
        
Args:
    train_df: The training split from split_train_test function
            must have FEATURE_COLUMNS and TARGET_COLUMN in order to train
Returns:
    Fitted HistGradientBoostingRegressor.
        
"""

    X_train = train_df[FEATURE_COLUMNS]
    Y_train = train_df[TARGET_COLUMN]

    model = HistGradientBoostingRegressor()
    model.fit(X_train, Y_train)

    return model



raw_df = get_tmax_data()
features_df = build_feature_table(raw_df)
train_df, test_df = split_train_test(features_df, cutoff_year=2015)

model = fit_model(train_df=train_df)
X_train = train_df[FEATURE_COLUMNS]
Y_train = train_df[TARGET_COLUMN]
print("Train R2:", model.score(X_train, Y_train))

X_test = test_df[FEATURE_COLUMNS]
Y_test = test_df[TARGET_COLUMN]
print("Test R2:", model.score(X_test, Y_test))