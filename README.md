# Financial Data Analysis and Portfolio Optimization

## Overview

This repository showcases my current work in upskilling with Python, focusing on financial data collection, cleaning, and analysis tailored for Quantitative Analyst skill development. The project emphasizes understanding financial data sources, extracting relevant information, and performing data-driven evaluations of stocks across various sectors. A key feature of this project is an interactive Dash application that visualizes stock metrics and sector comparisons, providing insights crucial for portfolio optimization and asset management.

## Credit
The original concept and code structure were inspired by the work of Faizan Ahmed, with significant modifications and enhancements made to tailor the analysis specifically for my learning and professional goals.

## Key Features
+ Data Collection and Cleaning: The foundation of this project is robust data handling. Using yfinance, the script fetches historical and financial data for a selection of stocks, ensuring data accuracy and completeness.
+ Sector Analysis: Financial metrics are categorized into Valuation, Profitability, Growth, and Performance. The data is cleaned to remove outliers, allowing for accurate sector-based comparisons.
+ Stock Grading System: Stocks are graded on financial metrics with a custom grading system that compares each stock's performance against sector averages. This system provides insights into each stock's strengths and weaknesses relative to its peers.
+ Comprehensive Reporting: Results are exported to CSV files, including detailed metric grades, sector comparisons, and overall stock ratings, making it easy to review and utilize the data in further analysis or presentations.
+ Interactive Dash Application: A user-friendly dashboard that visualizes detailed stock metrics and sector comparisons, enabling interactive exploration of stock performance data.

## Dash Application
The Dash application provides a dynamic interface for analyzing stock metrics and sector comparisons:

Dropdown Selection: Users can select stocks and grading categories from dropdown menus to filter and explore data specific to their interests.
Metric Comparison Graph: A bar chart visualizes the selected stock's metrics alongside sector benchmarks, highlighting how the stock compares to sector averages and percentile ranges.
Detailed Metric Grades Table: A data table displays detailed metric grades for the selected stock, with conditional formatting applied to profitability metrics for clear insights.

## Focus Areas
Data Collection: Understanding where and how to extract the necessary financial data for thorough analysis, ensuring reliability and relevance.
Data Cleaning: Emphasis on removing outliers and inconsistencies, which is critical for maintaining the integrity of the financial analysis.
Quantitative Analysis: Applying data-driven methodologies to evaluate stocks, focusing on metrics essential for Quantitative Analyst roles, such as valuations, growth projections, and performance evaluations.

## Tools and Libraries
Python: The primary programming language used for all data operations and analyses.
yfinance: For fetching historical and real-time market data from Yahoo Finance.
pandas: For data manipulation, cleaning, and analysis.
numpy: For numerical computations and statistical analysis.
tqdm: To provide progress tracking for long data-fetching processes.

## How to Use
Setup: Ensure you have Python installed along with the required libraries (yfinance, pandas, numpy, tqdm).
Run the Script: The main script data_analysis.py fetches stock data, cleans it, performs sector comparisons, and grades each stock based on predefined financial metrics.

Review the Output: CSV files will be generated in the specified directory, containing detailed analysis results which can be used for further financial modeling or decision-making.

## Future Work
Expansion to Other Data Sources: Integrating additional financial data sources for more comprehensive analysis.
Machine Learning Models: Incorporating predictive models to forecast stock performance based on historical data.
Enhanced Visualization: Adding data visualization capabilities to present insights more effectively.

### Contact
Feel free to reach out if you have any questions or suggestions regarding this project. I am actively seeking opportunities to leverage my skills in quantitative finance and data analysis.

 
