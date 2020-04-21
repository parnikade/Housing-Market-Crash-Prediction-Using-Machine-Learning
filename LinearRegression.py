import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import datetime as dt
import pandasql as ps


def main():
    # Data manipulation and processing
    # *************************************************************************

    df_house = pd.read_excel('House_price.xls').dropna(how='all').fillna(0)
    df_interest = pd.read_excel('interest_rate.xlsx',  parse_dates=True)
    df_total = pd.read_excel('house sold.xlsx', parse_dates=True)

    q = """
    Select observation_date, MORTGAGE30US
    From df_interest
    Where observation_date >= "1990-01-01"
    """

    df_in = ps.sqldf(q, locals())
    df_in.insert(0, 'ID', range(0, 0 + len(df_in)))
    df_in['observation_date'] = pd.to_datetime(df_in['observation_date'])
    df_in['year'] = df_in['observation_date'].dt.year
    df_in['month'] = df_in['observation_date'].dt.month

    q1 = """
    Select min(observation_date) as Date, month, year, max(MORTGAGE30US) as rate
    From df_in
    Group by year, month
    """
    df_final = ps.sqldf(q1, locals())
    df_final.to_csv('Final interest.csv')

    month_dict = {'Jan.': 1, 'Feb.': 2, 'March': 3, 'April': 4, 'May': 5,
                  'June': 6, 'July': 7, 'Aug.': 8, 'Sept.': 9, 'Oct.': 10,
                  'Nov.': 11, 'Dec.': 12}
    df_total.insert(0, 'ID', range(1, 1 + len(df_total)))
    # df_total.to_csv('total sold.csv')

    df_total['Month'].replace(to_replace=month_dict, inplace=True)
    df_total.to_csv('total sold.csv')
    # df_final['year'] = df_final['year'].astype(str)
    # df_final['year'] = df_final['year'].str[-2:]
    # df_final['Date'] = df_final[['month', 'year']].agg('-'.join, axis=1)

    df_house['month'] = df_house['Date'].dt.month
    df_house['year'] = df_house['Date'].dt.year

    df_CA = df_house[['Date', 'CA', 'month', 'year']]

    df_CA.to_csv('CA Avg.csv')

    q2 = """
    Select df_CA.Date, df_CA.month, df_CA.year, df_CA.CA as price, 
            df_final.rate, df_total.house_sold as total_house_sold
    From df_CA, df_final, df_total
    Where df_CA.month = df_final.month and df_CA.year = df_final.year 
     and df_CA.month = df_total.Month and df_CA.year = df_total.Year
    """
    df = ps.sqldf(q2, locals())
    df['period'] = df['month'].astype(str) + '/' + df['year'].astype(str)
    df.insert(0, 'ID', range(1, 1 + len(df)))
    df.to_csv('Price and rate.csv')
    print(df.columns)
    # *************************************************************************

    print('-' * 65)

    # Plotting of various graphs
    # *************************************************************************

    plt.scatter(df['year'], df['price'], color='red')
    plt.title('Date Vs Price', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.show()

    plt.scatter(df['rate'], df['price'], color='red')
    plt.title('Interest rate Vs Price', fontsize=14)
    plt.xlabel('Interest rate', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.show()

    plt.scatter(df['total_house_sold'], df['price'], color='red')
    plt.title('Houses sold Vs Price', fontsize=14)
    plt.xlabel('Houses sold', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.show()

    plt.scatter(df['year'], df['total_house_sold'], color='red')
    plt.title('Date Vs Houses sold', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Houses Sold', fontsize=14)
    plt.show()

    plt.scatter(df['year'], df['rate'], color='red')
    plt.title('Date vs Interest rate', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Interest rate', fontsize=14)
    plt.show()

    # *************************************************************************

    # Linear regression for house price prediction
    # *************************************************************************

    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(dt.datetime.toordinal)
    X = df[['Date', 'rate', 'total_house_sold']]
    Y = df['price']

    lin_reg = LinearRegression()
    lin_reg.fit(X, Y)
    lin_reg.predict(X)
    print(lin_reg.score(X, Y))

    print('Intercept: \n', lin_reg.intercept_)
    print('Coefficients: \n', lin_reg.coef_)
    print(lin_reg.predict(X))
    print('-' * 65)
    print(pd.DataFrame(zip(X.columns, lin_reg.coef_),
                 columns=['features', 'estimatedCofficients']))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=0)

    linreg = LinearRegression()
    linreg.fit(X_train, Y_train)
    Y_pred = linreg.predict(X_test)
    Y_train = linreg.predict(X_train)

    rmse = (np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    r2 = metrics.r2_score(Y_test, Y_pred)

    print('RMSE:', rmse)
    print('R square:', r2)

    print('-' * 65)
    print(linreg.score(X, Y))

    df_plot = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
    print(df_plot)

    plt.scatter(Y_test, Y_pred)
    axes = plt.gca()
    m, b = np.polyfit(Y_test, Y_pred, 1)
    X_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1]+100000, 100)
    plt.plot(X_plot, m * X_plot + b, '-')
    plt.xlabel('Prices')
    plt.ylabel('Predicted prices')
    plt.title('Prices vs Predicted Prices')
    plt.show()

    X_test_date = X_test['Date'].map(dt.datetime.fromordinal)
    df_plot2 = pd.DataFrame({'Date': X_test['Date'], 'Predicted': Y_pred})
    # print(df_plot2)
    plt.scatter(X_test_date, Y_pred)
    axes = plt.gca()
    m1, b1 = np.polyfit(X_test['Date'], Y_pred, 1)
    X_plot1 = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1]+1500, 100)
    plt.plot(X_plot1, m1 * X_plot1 + b1, '-')
    plt.xlabel('Date')
    plt.ylabel('Predicted prices')
    plt.title('Date vs Predicted Prices')
    plt.show()
    df_plot3 = pd.DataFrame({'Interest Rate': X_test['rate'],
                             'Predicted': Y_pred})
    print(df_plot3)
    plt.scatter(X_test['rate'], Y_pred)
    axes = plt.gca()
    m2, b2 = np.polyfit(X_test['rate'], Y_pred, 1)
    X_plot2 = np.linspace(axes.get_xlim()[0]-1, axes.get_xlim()[1]+1, 100)
    plt.plot(X_plot2, m2 * X_plot2 + b2, '-')
    plt.xlabel('Interest Rate')
    plt.ylabel('Predicted prices')
    plt.title('Interest Rate vs Predicted Prices')
    plt.show()

    df_plot3 = pd.DataFrame({'Houses Sold': X_test['total_house_sold'],
                             'Predicted': Y_pred})
    print(df_plot3)
    plt.scatter(X_test['total_house_sold'], Y_pred)
    axes = plt.gca()
    m3, b3 = np.polyfit(X_test['total_house_sold'], Y_pred, 1)
    X_plot3 = np.linspace(axes.get_xlim()[0]-1, axes.get_xlim()[1]+1000, 100)
    plt.plot(X_plot3, m3 * X_plot3 + b3, '-')
    plt.xlabel('Houses Sold')
    plt.ylabel('Predicted prices')
    plt.title('Houses Sold vs Predicted Prices')
    plt.show()


main()

