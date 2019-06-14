import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LinearRegression
import numpy as np
import re


class Model:
    def __init__(self):
        self.train_data = None
        self.predict_data = None
        self.train_set = []
        self.test_set = []
        self.load_data()
        self.split_data_set()
        self.feature_engineering()

    def load_data(self):
        train_col_name = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count',
                          'content']
        test_col_name = ['uid', 'mid', 'time', 'content']
        self.train_data = pd.read_table('./input/weibo_train_data.txt',
                                        names=train_col_name)
        self.predict_data = pd.read_table('./input/weibo_predict_data.txt', names=test_col_name)

    def split_data_set(self):
        self.train_data = self.train_data[0:4000]
        self.train_data['time'] = self.train_data['time'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))

        train_data1 = self.train_data[(self.train_data['time'] >= pd.to_datetime('20150201')) & (
                self.train_data['time'] <= pd.to_datetime('20150430'))]
        test_data1 = self.train_data[(self.train_data['time'] >= pd.to_datetime('20150501')) & (
                self.train_data['time'] <= pd.to_datetime('20150531'))]
        self.train_set.append(train_data1)
        self.test_set.append(test_data1)

        train_data2 = self.train_data[(self.train_data['time'] >= pd.to_datetime('20150301')) & (
                self.train_data['time'] <= pd.to_datetime('20150531'))]
        test_data2 = self.train_data[(self.train_data['time'] >= pd.to_datetime('20150601')) & (
                self.train_data['time'] <= pd.to_datetime('20150630'))]
        self.train_set.append(train_data2)
        self.test_set.append(test_data2)

        train_data3 = self.train_data[(self.train_data['time'] >= pd.to_datetime('20150401')) & (
                self.train_data['time'] <= pd.to_datetime('20150630'))]
        test_data3 = self.train_data[(self.train_data['time'] >= pd.to_datetime('20150701')) & (
                self.train_data['time'] <= pd.to_datetime('20150731'))]
        self.train_set.append(train_data3)
        self.test_set.append(test_data3)

    def feature_engineering(self):
        for data_frame in self.train_set:
            self.handle_user(data_frame)
            self.handle_date(data_frame)
            self.handle_content(data_frame)
            data_frame.drop(['uid', 'time', 'content'], axis=1, inplace=True)
        for data_frame in self.test_set:
            self.handle_user(data_frame)
            self.handle_date(data_frame)
            self.handle_content(data_frame)
            data_frame.drop(['uid', 'time', 'content'], axis=1, inplace=True)
        print("feature engineering ending...")

    """time's feature"""

    def handle_date(self, dataframe):
        dataframe['weekday'] = dataframe['time'].dt.weekday
        dataframe['hour_seg'] = dataframe['time'].dt.hour
        dataframe['isWeekend'] = dataframe['weekday'].apply(lambda x: 1 if x == 5 or x == 6 else 0)
        dr = pd.date_range(start='2015-02-01', end='2015-07-31')
        cal = calendar()
        holidays = cal.holidays(start=dr.min(), end=dr.max())
        dataframe['date'] = dataframe['time'].dt.date
        dataframe['holiday'] = dataframe['date'].apply(lambda x: 1 if x in holidays else 0)
        dataframe.drop('date', axis=1, inplace=True)

    """user's feature"""

    def handle_user(self, dataframe):
        grouped = dataframe.groupby(['uid'], as_index=False)
        # max feature
        max_forward_count = grouped[['forward_count']].max()
        max_forward_count['max_forward_count'] = max_forward_count['forward_count']
        max_forward_count.drop(['forward_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, max_forward_count, on='uid')

        max_comment_count = grouped[['comment_count']].max()
        max_comment_count['max_comment_count'] = max_comment_count['comment_count']
        max_comment_count.drop(['comment_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, max_comment_count, on='uid')

        max_like_count = grouped[['like_count']].max()
        max_like_count['max_like_count'] = max_like_count['like_count']
        max_like_count.drop(['like_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, max_like_count, on='uid')

        # min feature
        min_forward_count = grouped[['forward_count']].min()
        min_forward_count['min_forward_count'] = min_forward_count['forward_count']
        min_forward_count.drop(['forward_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, min_forward_count, on='uid')

        min_comment_count = grouped[['comment_count']].min()
        min_comment_count['min_comment_count'] = min_comment_count['comment_count']
        min_comment_count.drop(['comment_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, min_comment_count, on='uid')

        min_like_count = grouped[['like_count']].min()
        min_like_count['min_like_count'] = min_like_count['like_count']
        min_like_count.drop(['like_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, min_like_count, on='uid')

        # mean feature
        mean_forward_count = grouped[['forward_count']].mean()
        mean_forward_count['mean_forward_count'] = mean_forward_count['forward_count']
        mean_forward_count.drop(['forward_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, mean_forward_count, on='uid')

        mean_comment_count = grouped[['comment_count']].mean()
        mean_comment_count['mean_comment_count'] = mean_comment_count['comment_count']
        mean_comment_count.drop(['comment_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, mean_comment_count, on='uid')

        mean_like_count = grouped[['like_count']].mean()
        mean_like_count['mean_like_count'] = mean_like_count['like_count']
        mean_like_count.drop(['like_count'], axis=1, inplace=True)
        dataframe = pd.merge(dataframe, mean_like_count, on='uid')

        # max comment/like, forward/like
        dataframe['max_clr'] = dataframe['max_comment_count'].div(dataframe['max_like_count'])
        dataframe['max_cfl'] = dataframe['max_forward_count'].div(dataframe['max_like_count'])
        # min comment/like, forward/like
        dataframe['min_clr'] = dataframe['min_comment_count'].div(dataframe['min_like_count'])
        dataframe['min_cfl'] = dataframe['min_forward_count'].div(dataframe['min_like_count'])
        # mean comment/like, forward/like
        dataframe['mean_clr'] = dataframe['mean_comment_count'].div(dataframe['mean_like_count'])
        dataframe['mean_cfl'] = dataframe['mean_forward_count'].div(dataframe['mean_like_count'])

        max_like_mean = dict()
        max_comment_mean = dict()
        max_forward_mean = dict()
        for user in dataframe['uid']:
            count1 = 0
            count2 = 0
            count3 = 0
            u_info = dataframe[dataframe['uid'] == user]
            for i in u_info.index:
                f_info = dataframe[dataframe['uid'] == user]
                if u_info.loc[[i]]['like_count'].values[0] > f_info['mean_like_count'].values[0]:
                    count1 += 1

                if u_info.loc[[i]]['comment_count'].values[0] > f_info['mean_comment_count'].values[0]:
                    count2 += 1

                if u_info.loc[[i]]['forward_count'].values[0] > f_info['mean_forward_count'].values[0]:
                    count3 += 1
            mlm = count1 / u_info.ndim
            max_like_mean.update({user: mlm})

            mcm = count2 / u_info.ndim
            max_comment_mean.update({user: mcm})

            mfm = count3 / u_info.ndim
            max_forward_mean.update({user: mfm})

        max_like_mean_df = pd.DataFrame(data=np.transpose([list(max_like_mean.values()), list(max_like_mean.keys())]),
                                        columns=['max_like_mean', 'uid'])
        dataframe = pd.merge(dataframe, max_like_mean_df, on='uid')

        max_comment_mean_df = pd.DataFrame(
            data=np.transpose([list(max_comment_mean.values()), list(max_comment_mean.keys())]),
            columns=['max_comment_mean', 'uid'])
        dataframe = pd.merge(dataframe, max_comment_mean_df, on='uid')

        max_forward_mean_df = pd.DataFrame(
            data=np.transpose([list(max_forward_mean.values()), list(max_forward_mean.keys())]),
            columns=['max_forward_mean', 'uid'])
        dataframe = pd.merge(dataframe, max_forward_mean_df, on='uid')

        dataframe.fillna(0, inplace=True)
        dataframe.replace(-np.inf, 0, inplace=True)
        dataframe.replace(np.inf, 0, inplace=True)

    """content's feature"""

    def handle_content(self, dataframe):
        topic_pattern = re.compile('#.+#')
        reference_pattern = re.compile('【.+】')
        url_pattern = re.compile('[a-zA-z]+://[^\s]*')
        dataframe['if_topic'] = dataframe['content'].apply(lambda x: 1 if topic_pattern.findall(x) else 0)
        dataframe['if_aite'] = dataframe['content'].apply(lambda x: 1 if '@' in x else 0)
        dataframe['if_reference'] = dataframe['content'].apply(lambda x: 1 if reference_pattern.findall(x) else 0)
        dataframe['if_url'] = dataframe['content'].apply(lambda x: 1 if url_pattern.findall(x) else 0)
        dataframe['char_lens'] = dataframe['content'].apply(lambda x: len(x))


m = Model()
