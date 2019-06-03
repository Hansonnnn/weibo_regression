import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class EDA:
    def __init__(self):
        self.train_data = None
        self.load_data()

    def load_data(self):
        train_col_name = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count',
                          'content']
        self.train_data = pd.read_table('./input/weibo_train_data.txt',
                                        names=train_col_name)

    def plot(self):
        dataframe = self.train_data
        dataframe = dataframe[0:1000]
        fc = dataframe.groupby(['uid'])[['forward_count']].sum()

        plt.show()


e = EDA()
e.plot()
