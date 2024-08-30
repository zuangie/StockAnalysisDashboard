import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from tqdm import tqdm
import collections
import warnings

warnings.filterwarnings('ignore')

today_date = date.today().strftime("%m/%d/%y").replace('/', '.')

df = pd.read_csv(r"C:\Users\angela\Documents\Capped swix yf.csv")
tickers = df['tickers'].tolist()
tickers = tickers[:10]

allStockData = {}
sector_data = collections.defaultdict(lambda: collections.defaultdict(dict))
data_to_add = collections.defaultdict(list)
detailed_metric_data = []  # List to hold detailed metric grades for each stock
sector_comparative_data = []  # List to store sector comparative data



grading_metrics = {
    'Valuation': ['trailingPE', 'pegRatio', 'priceToSalesTrailing12Months', 'priceToBook', 'freeCashflow'],
    'Profitability': ['profitMargins', 'operatingMargins', 'grossMargins', 'returnOnEquity', 'returnOnAssets'],
    'Growth': ['earningsGrowth', 'revenueGrowth', 'earningsQuarterlyGrowth'],
    'Performance': ['52WeekChange', 'beta', 'shortPercentOfFloat']
}


def get_company_data(tickers):
    global allStockData
    dataframes = []
    start_date = "2010-01-01"
    end_date = "2024-07-31"

    print('\nFetching data from Yahoo Finance...\n')
    with tqdm(total=len(tickers)) as pbar:
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                info = stock.info
                data = {
                    'Ticker': ticker,
                    'Short Name': info.get('shortName', 'Unknown'),
                    'Sector': info.get('sector', 'Unknown')
                }
                # Include metrics for analysis
                data.update({metric: info.get(metric, np.nan) for category in grading_metrics.values() for metric in category})
                dataframes.append(pd.DataFrame([data]))
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

            pbar.update(1)

    allStockData = pd.concat(dataframes, ignore_index=True)


def remove_outliers(S, std):
    s1 = S[~((S - S.mean()).abs() > std * S.std())]
    return s1[~((s1 - s1.mean()).abs() > std * s1.std())]


def get_sector_data():
    global sector_data
    global allStockData

    sectors = allStockData['Sector'].unique()
    metrics = [metric for category in grading_metrics.values() for metric in category]

    for sector in sectors:
        rows = allStockData.loc[allStockData['Sector'] == sector]

        for metric in metrics:
            rows[metric] = pd.to_numeric(rows[metric], errors='coerce')
            data = remove_outliers(rows[metric], 2)

            sector_data[sector][metric]['Mean'] = data.mean(skipna=True)
            sector_data[sector][metric]['10Pct'] = data.quantile(0.1)
            sector_data[sector][metric]['90Pct'] = data.quantile(0.9)
            sector_data[sector][metric]['Std'] = np.std(data, axis=0) / 5

            # Collect sector comparative data for output
            sector_comparative_data.append({
                'Sector': sector,
                'Metric': metric,
                'Mean': sector_data[sector][metric]['Mean'],
                '10Pct': sector_data[sector][metric]['10Pct'],
                '90Pct': sector_data[sector][metric]['90Pct'],
                'Std': sector_data[sector][metric]['Std']
            })


def get_metric_val(ticker, metric_name):
    try:
        return float(allStockData.loc[allStockData['Ticker'] == ticker][metric_name].values[0])
    except (IndexError, ValueError):
        return np.nan


def convert_to_letter_grade(val):
    grade_scores = {
        'A+': 4.3, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0.0
    }

    sorted_grades = sorted(grade_scores.items(), key=lambda x: x[1], reverse=True)
    for grade, score in sorted_grades:
        if val >= score:
            return grade
    return 'C'



def get_metric_grade(sector, metric_name, metric_val):
    global sector_data

    lessThan = metric_name in ['trailingPE', 'pegRatio', 'priceToSalesTrailing12Months', 'priceToBook', 'freeCashflow',
                               'beta']
    grade_basis = '10Pct' if lessThan else '90Pct'
    start, change = sector_data[sector][metric_name][grade_basis], sector_data[sector][metric_name]['Std']

    grade_map = {
        'A+': 0, 'A': change, 'A-': change * 2, 'B+': change * 3, 'B': change * 4,
        'B-': change * 5, 'C+': change * 6, 'C': change * 7, 'C-': change * 8,
        'D+': change * 9, 'D': change * 10, 'D-': change * 11, 'F': change * 12
    }

    for grade, val in grade_map.items():
        comparison = start + val if lessThan else start - val
        if lessThan and metric_val < comparison:
            return grade
        if not lessThan and metric_val > comparison:
            return grade

    return 'C'


def collect_detailed_metric_data(ticker, sector):
    """
    Collects detailed information on each metric within each grading category, including the
    stock's metric value, sector comparisons, and assigned grades.
    """
    global detailed_metric_data
    # Fetch the short name for the given ticker
    stock_name = allStockData.loc[allStockData['Ticker'] == ticker, 'Short Name'].values[0]

    for category, metrics in grading_metrics.items():
        for metric in metrics:
            metric_val = get_metric_val(ticker, metric)
            sector_mean = sector_data[sector][metric].get('Mean', np.nan)
            sector_10pct = sector_data[sector][metric].get('10Pct', np.nan)
            sector_90pct = sector_data[sector][metric].get('90Pct', np.nan)
            grade = get_metric_grade(sector, metric, metric_val)

            detailed_metric_data.append({
                'Category': category,
                'Ticker': ticker,
                'Short Name': stock_name,
                'Sector': sector,
                'Metric': metric,
                'Metric Value': metric_val,
                'Sector Mean': sector_mean,
                'Sector 10Pct': sector_10pct,
                'Sector 90Pct': sector_90pct,
                'Assigned Grade': grade
            })

def get_category_grades(ticker, sector):
    global grading_metrics

    grade_scores = {
        'A+': 4.3, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0.0
    }

    category_grades = {}

    for category, metrics in grading_metrics.items():
        total_points = 0
        valid_metrics = 0
        for metric in metrics:
            metric_val = get_metric_val(ticker, metric)
            if not np.isnan(metric_val):
                grade = get_metric_grade(sector, metric, metric_val)
                total_points += grade_scores[grade]
                valid_metrics += 1

        avg_grade = total_points / valid_metrics if valid_metrics > 0 else 0
        closest_grade = min(grade_scores.keys(), key=lambda g: abs(grade_scores[g] - avg_grade))
        category_grades[category] = closest_grade

    return category_grades


def get_stock_rating(category_grades):
    grade_scores = {
        'A+': 4.3, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0.0
    }

    total_points = sum(grade_scores[grade] for grade in category_grades.values())
    avg_points = total_points / len(category_grades) if category_grades else 0
    overall_grade = min(grade_scores.keys(), key=lambda g: abs(grade_scores[g] - avg_points))

    return overall_grade


def get_stock_rating_data():
    global data_to_add
    global allStockData

    print('\nCalculating Stock Ratings...\n')
    with tqdm(total=allStockData.shape[0]) as pbar:
        for index, row in allStockData.iterrows():
            ticker, sector = row['Ticker'], row['Sector']

            category_grades = get_category_grades(ticker, sector)
            overall_rating = get_stock_rating(category_grades)


            collect_detailed_metric_data(ticker, sector)

            data_to_add['Overall Rating'].append(overall_rating)
            data_to_add['Valuation Grade'].append(category_grades.get('Valuation', 'F'))
            data_to_add['Profitability Grade'].append(category_grades.get('Profitability', 'F'))
            data_to_add['Growth Grade'].append(category_grades.get('Growth', 'F'))
            data_to_add['Performance Grade'].append(category_grades.get('Performance', 'F'))

            pbar.update(1)


def export_to_csv(filename, filepath=''):
    global allStockData

    comparison_df = pd.DataFrame(data_to_add)
    allStockData = pd.concat([allStockData, comparison_df], axis=1)

    allStockData['Overall Rating'] = data_to_add['Overall Rating']
    allStockData['Valuation Grade'] = data_to_add['Valuation Grade']
    allStockData['Profitability Grade'] = data_to_add['Profitability Grade']
    allStockData['Growth Grade'] = data_to_add['Growth Grade']
    allStockData['Performance Grade'] = data_to_add['Performance Grade']

    # comparison_df = pd.DataFrame(data_to_add)
    # allStockData = pd.concat([allStockData, comparison_df], axis=1)

    ordered_columns = [
        'Ticker','Short Name', 'Overall Rating', 'Sector', 'Valuation Grade', 'Profitability Grade',
        'Growth Grade', 'Performance Grade', *grading_metrics['Valuation'],
        *grading_metrics['Profitability'], *grading_metrics['Growth'], *grading_metrics['Performance'],
        *comparison_df.columns  # Include all comparison columns
    ]

    stock_csv_data = allStockData[ordered_columns]

    # Save the main stock data to CSV
    stock_csv_path = f"{filepath}{filename}_StockData.csv"
    stock_csv_data.to_csv(stock_csv_path, index=False)

    # Save the detailed metric grades data to CSV
    detailed_metric_df = pd.DataFrame(detailed_metric_data)
    detailed_metric_path = f"{filepath}{filename}_MetricGrades.csv"
    detailed_metric_df.to_csv(detailed_metric_path, index=False)

    # Save the sector comparative data to CSV
    sector_comparative_df = pd.DataFrame(sector_comparative_data)
    sector_comparative_path = f"{filepath}{filename}_SectorComparativeData.csv"
    sector_comparative_df.to_csv(sector_comparative_path, index=False)

    print(f'\nSaved Stock Data as: {stock_csv_path}')
    print(f'Saved Metric Grades as: {detailed_metric_path}')
    print(f'Saved Sector Comparative Data as: {sector_comparative_path}')



filename = "datapack_.csv"
filepath = r"C:\Users\angela\Documents\name"  # Corrected path with raw string

# Sample list of tickers



get_company_data(tickers)
get_sector_data()
get_stock_rating_data()
export_to_csv(filename=filename, filepath=filepath)
