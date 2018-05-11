# Basics
import numpy as np
import pandas as pd
from datetime import datetime
from time import time

# Machine learning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
#from NN import LSTM

## Plotting
#import seaborn as sb
#import matplotlib.pyplot as plt
#import matplotlib as mat
#import plotly
#from plotly.graph_objs import Scatter, Layout



# Loading and preprocessing data
#-------------------------------------------------------------------------------
# Import data

CSV_URL = 'https://raw.githubusercontent.com/VioletaStoyanova/data602-finalproject/master/bitcoin_dataset.csv?token=AioA2smxqhyuCAVDnBPdVBfZ2dvydIVIks5a_y9CwA%3D%3D'


data = pd.read_csv(CSV_URL)

# Convert date string to datetime
data.Date = pd.to_datetime(data.Date)

# Clean up feature names
cols = []
for col in data:
    col = col[4:] if col != 'Date' else col
    cols.append(col)
data.columns = cols

# There are a couple of NaN values in 'trade_volume'. Backfill.
data = data.fillna(method='backfill')


# Plotting
#-------------------------------------------------------------------------------
def feature_vs_date(data):
    # Plot features vs. date
    X = data.Date
    Y = data.drop(['Date'], axis=1)

    plots = {}
    for i, col in enumerate(Y):
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(X, Y[col])
        ax.set_title(f'Date vs {col}')
        ax.set_xlabel('Date')
        ax.set_ylabel(col)
        plots[col] = f
    return plots


def feature_vs_price(data):
    # Plot features vs. price
    X = data.drop(['market_price', 'Date'], axis=1)
    Y = data.market_price

    plots = {}
    for i, col in enumerate(X):
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(X[col], Y, 'ko', markersize=2)
        ax.set_title(f'BTC Price vs. {col}')
        ax.set_ylabel('BTC Price')
        ax.set_xlabel(col)
        plots[col] = f
    return plots


def heatmap(data):
    # Plot linear correlation between features
    fig, ax  = plt.subplots()
    corr = data.drop(['Date'],axis=1).corr()
    g = sb.heatmap(corr, yticklabels=corr.columns, ax=ax)
    return fig


def draw_all_plots(data):
    feature_vs_date(data)
    feature_vs_price(data)
    heatmap(data)
    plt.show()


def plot_predictions(X, actual, prediction, title, target, x_label,
                     filename = 'default', auto_open = False):


    filename = f'{title}' if filename == 'default' else filename
    actual = Scatter(x = X, y = actual, name = 'Actual')
    prediction = Scatter(x = X, y = prediction, name = 'Prediction')

    x_axis_template = dict(title = x_label)
    y_axis_template = dict(title = target)

    layout = Layout(
        xaxis = x_axis_template,
        yaxis = y_axis_template,
        title = title
    )

    filename = filename + '.html'
    plot_data = [actual, prediction]
    plotly.offline.plot({'data': plot_data,
                         'layout': layout},
                          filename = filename,
                          auto_open = auto_open)



# Feature Engineering
#-------------------------------------------------------------------------------

# Define X and y for splitting into train/test sets
X = data.drop(['Date', 'market_price'], axis=1)
y = data.market_price

# Convert market_price to binary feature for classification:
# price increase = 1, price decrease = 0
f = lambda x: 1 if x >= 0 else 0
y_bin = y.diff().map(f)

# Basic train/test split
split_percent = .8
def split_data(X, y, split_percent):
    split_ind = int(len(X)*split_percent)
    X_train = X.iloc[:split_ind, :]
    y_train = y.iloc[:split_ind]
    X_test = X.iloc[split_ind:, :]
    y_test = y.iloc[split_ind:]
    return X_train, y_train, X_test, y_test

# Split dates seperately to correspond to train/test sets
def split_dates(dates, split_percent):
    split_ind = int(len(X)*split_percent)
    dates_train = dates.iloc[:split_ind]
    dates_test = dates.iloc[split_ind:]
    return dates_train, dates_test

# Regression data
X_train_r, y_train_r, X_test_r, y_test_r = split_data(X, y, split_percent)

# Classification data
X_train_c, y_train_c, X_test_c, y_test_c = split_data(X, y_bin, split_percent)

# Get sliced dates
dates_train, dates_test = split_dates(data.Date, split_percent)




# Machine Learning
#-------------------------------------------------------------------------------

# Initialize classification models
logr = LogisticRegression()
rfc = RandomForestClassifier()
svc = SVC()
classification_models = {'Logistic Regression':logr,
                         'Random Forest Classifier':rfc,
                         'Support Vector Classifier': svc}

# Initialize regression models
linr = LinearRegression()
rfr = RandomForestRegressor()
svr = SVR()
regression_models = {'Linear Regression':linr,
                     'Random Forest Regressor':rfr,
                     'Support Vector Regressor': svr}


train = True
if train:
    # Train Regression models
    print('------Regression------')
    for model_name, model in regression_models.items():
        t = time()
        model.fit(X_train_r, y_train_r)
        t = time() - t
        print(f'Trained {model_name} in {t} sec')
        print(f'Score: {model.score(X_test_r, y_test_r)}')

    # Train Classification models
    print('------Classification------')
    for model_name, model in classification_models.items():
        t = time()
        model.fit(X_train_c, y_train_c)
        t = time() - t
        print(f'Trained {model_name} in {t} sec')
        print(f'Score: {model.score(X_test_c, y_test_c)}')



#plot = True
#if plot:
    ## Plot regression predictions
    #for model_name, model in regression_models.items():
        #actual = y_test_r
        #prediction = model.predict(X_test_r)
        #title = f'{model_name} predictions vs. Actual: Regression '
        #target = 'Market Price'
        #x_label = 'Date'
        #filename = f'.{model_name}'
        #plot_predictions(dates_test, actual, prediction, title, target, x_label,
                         #filename = filename)

    
    ## Plot classification predictions
    #for model_name, model in classification_models.items():
        #actual = y_test_c
        #prediction = model.predict(X_test_c)
        #title = f'{model_name} predictions vs. Actual: Classification'
        #target = 'Market Price Change'
        #x_label = 'Date'
        #filename = f'.{model_name}'
        #plot_predictions(dates_test, actual, prediction, title, target, x_label,
                         #filename = filename)
