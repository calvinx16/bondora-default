"""Uni project to predict loan default rates"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_data():
    loan_df = pd.read_csv(os.path.join(DATA_PATH, 'LoanData_10-4-2022.csv'))
    stock_df = pd.read_excel(os.path.join(DATA_PATH, 'Stock_market.xlsx'))
    inflation_df = pd.read_excel(os.path.join(DATA_PATH, 'CPI.xlsx'))
    unemployment_df = pd.read_excel(os.path.join(DATA_PATH, 'Unemployment.xlsx'))
    return loan_df, stock_df, inflation_df, unemployment_df


def preprocess_loans(df):
    df['LoanDate'] = pd.to_datetime(df['LoanDate'])
    df = df[(df['Country'] == 'EE') & (df['LoanDate'].dt.year.between(2012, 2021))]
    df = df[df['Status'].isin(['Repaid', 'Late'])]
    df['Bad_loans'] = np.where(df['Status'] == 'Late', 1, 0)
    df['ExpectedLoss'].fillna(0, inplace=True)
    return df


def merge_macro_data(loans, stock, cpi, unemployment):
    for df, date_col in [(stock, 'Date'), (cpi, 'Date'), (unemployment, 'Period')]:
        df[date_col] = pd.to_datetime(df[date_col])
        df['grouper'] = df[date_col].dt.to_period('M')
    loans['grouper'] = loans['LoanDate'].dt.to_period('M')
    merged = loans.merge(stock, on='grouper', how='left') \
        .merge(cpi, on='grouper', how='left') \
        .merge(unemployment, on='grouper', how='left')
    return merged.drop(columns=['grouper', 'Date', 'Period'], errors='ignore')


def plot_macro(stock, cpi, unemployment):
    stock['Date'] = pd.to_datetime(stock['Date'])
    cpi['Date'] = pd.to_datetime(cpi['Date'])
    unemployment['Period'] = pd.to_datetime(unemployment['Period'])

    plt.plot(stock['Date'], stock['Stock_Price'])
    plt.title('OMX Tallinn Index')
    plt.show()

    plt.plot(cpi['Date'], cpi['CPI'])
    plt.title('CPI Over Time')
    plt.show()

    plt.plot(unemployment['Period'], unemployment['Unemployment rate (in %)'])
    plt.title('Unemployment Rate')
    plt.show()


def run_random_forest(df, features):
    X = df[features]
    y = df['Bad_loans'].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    model = RandomForestClassifier(n_estimators=10, random_state=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Features: {features}")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def main():
    loans, stock, cpi, unemployment = load_data()
    loans = preprocess_loans(loans)
    df = merge_macro_data(loans, stock, cpi, unemployment)

    # Run models
    run_random_forest(df, ['ExpectedLoss'])
    run_random_forest(df, ['Stock_Price', 'CPI', 'Unemployment rate (in %)'])


if __name__ == "__main__":
    main()
