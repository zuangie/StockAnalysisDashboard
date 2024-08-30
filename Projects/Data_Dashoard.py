import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
from dash.dash_table.Format import Format, Scheme

# Load data from CSV files
# stockdata_df = pd.read_csv(r"C:\Users\angela\Documents\namedatapack_.csv_StockData.csv")
metric_grades_df = pd.read_csv(r"C:\Users\angela\Documents\namedatapack_.csv_MetricGrades.csv")
sector_comparative_df = pd.read_csv(r"C:\Users\angela\Documents\namedatapack_.csv_SectorComparativeData.csv")

metric_grades_df['Metric Value'] =metric_grades_df['Metric Value'].round(1)
metric_grades_df['Sector Mean'] = metric_grades_df['Sector Mean'].round(1)
metric_grades_df['Sector 10Pct'] = metric_grades_df['Sector 10Pct'].round(1)
metric_grades_df['Sector 90Pct'] = metric_grades_df['Sector 90Pct'].round(1)


# Define profitability metrics to be converted to percentages
profitability_metrics = ['profitMargins', 'operatingMargins', 'grossMargins', 'returnOnEquity', 'returnOnAssets']

# # Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout of the dashboard
app.layout = html.Div([
    html.H1("Stock Grading Dashboard", style={'text-align': 'center'}),

    # Dropdown for selecting a stock short name
    html.Div([
        html.Label('Select a Stock:'),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': name, 'value': name} for name in metric_grades_df['Short Name'].unique()],
            value=metric_grades_df['Short Name'].unique()[0]
        ),
    ], style={'width': '50%', 'display': 'inline-block'}),

    # Dropdown for selecting a grading category
    html.Div([
        html.Label('Select a Grading Category:'),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': category, 'value': category} for category in metric_grades_df['Category'].unique()],
            value=metric_grades_df['Category'].unique()[0]
        ),
    ], style={'width': '50%', 'display': 'inline-block'}),

    # Graph to display metric values and sector comparison
    dcc.Graph(id='metric-comparison-graph'),

    # Data table to show detailed metric grades for the selected stock
    dash_table.DataTable(
        id='metric-details-table',
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed)   # Default formatting for all numeric values
            }
            for i in metric_grades_df.columns if i != 'Sector'  # Exclude the 'Sector' column
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Metric} = "profitMargins" || {Metric} = "operatingMargins" || {Metric} = "grossMargins" || {Metric} = "returnOnEquity" || {Metric} = "returnOnAssets"',
                    'column_id': ['Metric Value', 'Sector Mean', 'Sector 10Pct', 'Sector 90Pct']
                },
                'format': Format(precision=1, scheme=Scheme.percentage)
                # Apply percentage formatting only to profitability metrics
            }
        ]
    )
])

# Callback to update the graph based on selected stock and category
@app.callback(
    Output('metric-comparison-graph', 'figure'),
    [Input('ticker-dropdown', 'value'),
     Input('category-dropdown', 'value')]
)
def update_graph(selected_ticker, selected_category):
    # Filter data for the selected stock and category
    filtered_df = metric_grades_df[(metric_grades_df['Short Name'] == selected_ticker) &
                                   (metric_grades_df['Category'] == selected_category)]

    # Create a bar graph showing stock metric values and sector benchmarks
    fig = px.bar(filtered_df, x='Metric', y=['Metric Value', 'Sector Mean', 'Sector 10Pct', 'Sector 90Pct'],
                 barmode='group', title=f"Comparison for {selected_ticker} in {selected_category}",
                 labels={'value': 'Value', 'variable': 'Metric Type'})

    fig.update_layout(legend_title_text='Comparison Type', xaxis_title='Metrics', yaxis_title='Values')
    return fig

# Callback to update the detailed metric grades table
@app.callback(
    Output('metric-details-table', 'data'),
    [Input('ticker-dropdown', 'value'),
     Input('category-dropdown', 'value')]
)
def update_table(selected_ticker, selected_category):
    # Filter data for the selected stock and category
    filtered_df = metric_grades_df[(metric_grades_df['Short Name'] == selected_ticker) &
                                   (metric_grades_df['Category'] == selected_category)]
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)