#Creating a Sector Rotation Strategy Using Predictive Analytics
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pandas_datareader import data as pdr
from tqdm import tqdm




start = '2010-06-30'
end = '2024-07-30'
#stocks data list
SAVIT40 = pd.read_csv(r"C:\Users\angela\Documents\SAVIT40 Index.csv")
SAVIT40['Date'] = pd.to_datetime(SAVIT40['Date'])
SAVIT40.set_index('Date', inplace=True)
SAVIT40.columns = ['SAVIT40']
print(SAVIT40)

sa_index = yf.download('STXSWX.JO', start = start, end = end )
sa_index = sa_index[['Close']]
sa_index.columns = ['SA_Index_Close']
print(sa_index)
#Get economic data

# df = pd.read_csv(r"C:\Users\angela\Documents\Capped swix yf.csv")


ticker_to_name = {
    'STXSWX.JO': 'SWIX Top 40',
    'STXRAF.JO': 'RAF 40',
    'STXRES.JO': 'Resources 10',
    'STXFIN.JO': 'Financial 15',
    'STXPRO.JO': 'Property Index',
    'STXIND.JO': 'Industrial 25',
    'STXGOV.JO': 'Government Bond'
}

# Download sector index data
tickers = ['STXSWX.JO', 'STXRAF.JO', 'STXRES.JO', 'STXFIN.JO', 'STXPRO.JO', 'STXIND.JO', 'STXGOV.JO']
sectors = yf.download(tickers, start=start, end=end)
sectors = sectors[['Close']]

# Flatten the multi-level columns (e.g., ('Close', 'STXFIN.JO') -> 'STXFIN.JO')
sectors.columns = sectors.columns.get_level_values(1)

# Rename the columns using the mapping from ticker to full stock name
sectors.columns = [f"{ticker_to_name[ticker]}" for ticker in sectors.columns]
print(sectors.head())

# Reset index to ensure compatibility with macro_data (if necessary)
sectors_flat = sectors.copy()
sectors_flat.reset_index(inplace=True)
sectors_flat.set_index('Date', inplace=True)


FRED_INDICATORS = ["ZAFGDPNQDSMEI",         #GDP Q/Q
                   "IRSTCI01ZAM156N",       #Intrest rates
                   "ZAFCPALTT01IXNBM",      #CPI SA
                   "IRLTLT01ZAM156N"        #10YRGovBond Yiled SA
                   ]


# update period for each ind (Y=Yearly, Q=Quarterly, M=Monthly, W=Weekly, D=Daily)
INDICATORS_PERIODS = {'ZAFGDPNQDSMEI': 'Q',  # 1. Growth
                      # 2. Intrest Rates
                      'IRSTCI01ZAM156N': 'M',
                      # 3. Inflation
                      'ZAFCPALTT01IXNBM': 'M',
                      # 4. 10 Year Bond yield
                      'SAVIT40': 'M',
                      # 3. Inflation
                      'sa_index': 'M',
                      # 4. 10 Year Bond yield
                      }
for ticker in sectors_flat.columns:
    INDICATORS_PERIODS[ticker] = 'D'

end = dt.datetime(2023,12,31)
start = dt.datetime(year=end.year-10, month=end.month, day=end.day)
macro_indicators = dict()
tq_fred = tqdm(FRED_INDICATORS)

# get the stats from FRED database (with Pandas Datareader API)
tq_fred.set_description('Downloading stats from FRED:')
for indicator in tq_fred:
  # macro_indicators[indicator] = pdr.DataReader(indicator, "fred", start=start, timeout=90)
  macro_indicators[indicator] = pdr.FredReader(indicator, start=start).read()


macro_data = pd.concat(macro_indicators.values(), axis=1)
macro_data.columns = macro_indicators.keys()

macro_data = macro_data.join(SAVIT40, how='left', rsuffix='SAVIT40')
macro_data = macro_data.join(sectors_flat, how='left')
# macro_data = macro_data.join(sa_index, how='left', rsuffix='sa_index')

#renaming
macro_data.rename(columns={
    "ZAFGDPNQDSMEI": "SA_GDP_QQ",
    "IRSTCI01ZAM156N": "REPO_Rate",
    "ZAFCPALTT01IXNBM": "CPI_SA",
    "IRLTLT01ZAM156N": "10YR_SAGOV_Yield"},
    inplace= True)



macro_indicators_dict = copy.deepcopy(macro_data)
def get_macro_shift_transformation(macro_indicators_dict):
        """Add shifted (growth) values to the macro_indicators_dict before joining them together, remove non-stationary time series."""

        # Transformations based on the frequency of indicators
        HISTORICAL_PERIODS_DAYS = [1, 3, 7, 30, 90, 365]

        # Different types of transformations for daily, weekly, monthly, quarterly, yearly indicators
        # Applying transformations
        for ind in macro_indicators_dict.columns:
            for period in HISTORICAL_PERIODS_DAYS:
                macro_indicators_dict[f'{ind}_Shift_{period}'] = macro_indicators_dict[ind].pct_change(periods=period)

        # Removing non-stationary time series if all values are NaN after transformation
        for key in list(macro_indicators_dict.columns):
            if macro_indicators_dict[key].isnull().all():
                macro_indicators_dict.drop(columns=[key], inplace=True)

        return macro_indicators_dict


transformed_macro_data = get_macro_shift_transformation(macro_indicators_dict)
transformed_macro_data.
# file_name = r'C:\Users\angela\Downloads\macro data.xlsx'
# transformed_macro_data.to_excel(file_name)

# Handle missing data by forward and backward filling
# Handle missing data by forward and backward filling
# Check the columns to identify the date column
data_filled = transformed_macro_data.ffill().bfill()
data_filled.reset_index(inplace=True)

# Now, the date column should be present as the first column, likely named 'index' or something similar
date_column_name = data_filled.columns[0]  # This should now reference the correct column for dates

# Separate the target columns (sectors) and features (macroeconomic indicators and their shifts)
sectors = [col for col in data_filled.columns if 'Shift' not in col and col not in [date_column_name, 'SA_GDP_QQ', 'REPO_Rate', 'CPI_SA', '10YR_SAGOV_Yield']]
features = [col for col in data_filled.columns if col not in sectors + [date_column_name]]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_filled[features])

# Convert back to a DataFrame for easier handling
scaled_features_df = pd.DataFrame(scaled_features, columns=features)

# Combine scaled features with original (non-scaled) target sector data
final_data = pd.concat([data_filled[date_column_name], scaled_features_df, data_filled[sectors]], axis=1)


# Set the "DATE" column as the index
final_data.set_index('DATE', inplace=True)
# Save the final data to an Excel file
file_name = r'C:\Users\angela\Downloads\final_macro.xlsx'
final_data.to_excel(file_name, index=False)

# Save the final data to an Excel file
file_name = r'C:\Users\angela\Downloads\final_macro.xlsx'
final_data.to_excel(file_name)

# Display the first few rows of the final processed data
print(final_data.head())

file_name = r'C:\Users\angela\Downloads\final macro.xlsx'
final_data.to_excel(file_name)

print(final_data.columns)

# Identify the features and targets
features = [col for col in final_data.columns if 'Shift' in col]
targets = [col for col in final_data.columns if
           col not in features + ['Date']]  # Replace 'Date' with your actual date column name if different

# Loop through each sector (target) to perform analysis
for target in targets:
    print(f"Analyzing sector: {target}")

    # Define X and y
    X = final_data[features]
    y = final_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Feature importance
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {target}')
    plt.gca().invert_yaxis()
    plt.show()

    # Display the top 10 important features
    print(feature_importance_df.head(10))
    print("\n")