import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import pandas as pd


def main():

    df = pd.read_csv('Price and rate and houses sold.csv', parse_dates=True)
    df.drop(df.index[0])

    prices = np.array(df['price'])
    num_of_houses_sold = np.array(df['total_house_sold'])
    rate = np.array(df['rate'])
    dates = np.array(df['period'])

    diff_percentages = 100.0 * np.diff(prices) / prices[:-1]
    diff_percentages = np.append([0], diff_percentages)

    data = np.column_stack([diff_percentages, prices])

    hmm = GaussianHMM(n_components=15, covariance_type='tied', n_iter=100000,
                      algorithm='viterbi', random_state=False)

    hmm.fit(data)
    pred_count = 12

    num_samples = len(data)
    samples, _ = hmm.sample(num_samples + pred_count)
    print(samples)

    plt.figure()
    plt.xlabel('Days starting from Jan 1990')
    plt.ylabel('House prices predicted and actual')
    plt.title('Days vs Prices')

    plt.plot(np.arange(num_samples + pred_count), samples[:, 1], 'r--',
             np.arange(num_samples), prices[:num_samples], 'b-')
    plt.ylim(ymin=0)
    plt.show()


main()


