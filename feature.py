import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LinearRegression
from evaluate import score
import numpy as np
import re
from jieba import analyse

import threading


class Feature:
    def __init__(self):
        self.train_data = None
        self.predict_data = None
        self.train_set = []
        self.test_set = []
        self.predict_user_feature = []
        self.tfidf = analyse.extract_tags
        self.load_data()
        self.split_data_set()
        # 提取一次top20关键词就可以了，之后就可注掉
        self.keywords = self.get_topk_word(self.train_data)
        self.feature_engineering()

    def load_data(self):
        train_col_name = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count',
                          'content']
        predict_col_name = ['uid', 'mid', 'time', 'content']
        self.train_data = pd.read_table('./input/weibo_train_data.txt',
                                        names=train_col_name)
        self.predict_data = pd.read_table('./input/weibo_predict_data.txt', names=predict_col_name)

    def split_data_set(self):
        self.predict_data['time'] = self.predict_data['time'].apply(
            lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
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
        def train_feature(data_frame, i):
            print('==============current thread name is {%s} ,{%d}' % (threading.current_thread().name, i))

            data_frame[['like_count', 'forward_count', 'comment_count']].fillna(0, inplace=True)
            data_frame['content'].replace(np.nan, " ", inplace=True)
            self.handle_date(data_frame)
            self.handle_content(data_frame)
            data_frame = self.handle_train_user(data_frame)
            print('==========train_set output to csv file ===========')
            data_frame.to_csv('train_set' + str(i) + '.csv', index=False)

        def test_feature(data_frame, i):
            print('==============current thread name is {%s} ,{%d}' % (threading.current_thread().name, i))
            data_frame[['like_count', 'forward_count', 'comment_count']].fillna(0, inplace=True)
            data_frame['content'].replace(np.nan, " ", inplace=True)
            self.handle_date(data_frame)
            self.handle_content(data_frame)
            data_frame = self.handle_train_user(data_frame)
            print('==========test_set output to csv file ===========')
            data_frame.to_csv('test_set' + str(i) + '.csv', index=False)

        def predict_feature(data_frame):
            print('==============current thread name is {%s}' % threading.current_thread().name)
            data_frame['content'].replace(np.nan, " ", inplace=True)
            self.handle_date(data_frame)
            self.handle_content(data_frame)
            data_frame = self.handle_predict_user(data_frame)
            print('==========predict_set output to csv file ===========')
            data_frame.to_csv('predict.csv', index=False)

        t1 = threading.Thread(target=train_feature, name='train_set0', args=(self.train_set[0], 0))
        t2 = threading.Thread(target=train_feature, name='train_set1', args=(self.train_set[1], 1))
        t3 = threading.Thread(target=train_feature, name='train_set2', args=(self.train_set[2], 2))
        t4 = threading.Thread(target=test_feature, name='test_set0', args=(self.test_set[0], 0))
        t5 = threading.Thread(target=test_feature, name='test_set1', args=(self.test_set[1], 1))
        t6 = threading.Thread(target=test_feature, name='test_set2', args=(self.test_set[2], 2))
        t7 = threading.Thread(target=predict_feature, name='predict', args=(self.predict_data,))
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
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
        dataframe.drop('time', axis=1, inplace=True)

    """user's feature"""

    def handle_train_user(self, dataframe):
        user_info = self.group_user(dataframe)
        for df in user_info:
            dataframe = pd.merge(dataframe, df, on='uid')
        # max comment/like, forward/like
        dataframe['max_clr'] = dataframe['max_comment_count'].div(dataframe['max_like_count'])
        dataframe['max_cfl'] = dataframe['max_forward_count'].div(dataframe['max_like_count'])
        # min comment/like, forward/like
        dataframe['min_clr'] = dataframe['min_comment_count'].div(dataframe['min_like_count'])
        dataframe['min_cfl'] = dataframe['min_forward_count'].div(dataframe['min_like_count'])
        # mean comment/like, forward/like
        dataframe['mean_clr'] = dataframe['mean_comment_count'].div(dataframe['mean_like_count'])
        dataframe['mean_cfl'] = dataframe['mean_forward_count'].div(dataframe['mean_like_count'])

        dataframe.fillna(0, inplace=True)
        dataframe.replace(-np.inf, 0, inplace=True)
        dataframe.replace(np.inf, 0, inplace=True)
        return dataframe

    def handle_predict_user(self, dataframe):
        user_info = self.group_user(self.train_data)
        for df in user_info:
            dataframe = pd.merge(dataframe, df, on='uid')
        # max comment/like, forward/like
        dataframe['max_clr'] = dataframe['max_comment_count'].div(dataframe['max_like_count'])
        dataframe['max_cfl'] = dataframe['max_forward_count'].div(dataframe['max_like_count'])
        # min comment/like, forward/like
        dataframe['min_clr'] = dataframe['min_comment_count'].div(dataframe['min_like_count'])
        dataframe['min_cfl'] = dataframe['min_forward_count'].div(dataframe['min_like_count'])
        # mean comment/like, forward/like
        dataframe['mean_clr'] = dataframe['mean_comment_count'].div(dataframe['mean_like_count'])
        dataframe['mean_cfl'] = dataframe['mean_forward_count'].div(dataframe['mean_like_count'])
        return dataframe

    def group_user(self, dataframe):
        user_infos = []
        grouped = dataframe.groupby(['uid'], as_index=False)
        # max feature
        max_forward_count = grouped[['forward_count']].max()
        max_forward_count['max_forward_count'] = max_forward_count['forward_count']
        max_forward_count.drop(['forward_count'], axis=1, inplace=True)
        user_infos.append(max_forward_count)
        # dataframe = pd.merge(dataframe, max_forward_count, on='uid')

        max_comment_count = grouped[['comment_count']].max()
        max_comment_count['max_comment_count'] = max_comment_count['comment_count']
        max_comment_count.drop(['comment_count'], axis=1, inplace=True)
        user_infos.append(max_comment_count)
        # dataframe = pd.merge(dataframe, max_comment_count, on='uid')

        max_like_count = grouped[['like_count']].max()
        max_like_count['max_like_count'] = max_like_count['like_count']
        max_like_count.drop(['like_count'], axis=1, inplace=True)
        user_infos.append(max_like_count)
        # dataframe = pd.merge(dataframe, max_like_count, on='uid')

        # min feature
        min_forward_count = grouped[['forward_count']].min()
        min_forward_count['min_forward_count'] = min_forward_count['forward_count']
        min_forward_count.drop(['forward_count'], axis=1, inplace=True)
        user_infos.append(min_forward_count)
        # dataframe = pd.merge(dataframe, min_forward_count, on='uid')

        min_comment_count = grouped[['comment_count']].min()
        min_comment_count['min_comment_count'] = min_comment_count['comment_count']
        min_comment_count.drop(['comment_count'], axis=1, inplace=True)
        user_infos.append(min_comment_count)
        # dataframe = pd.merge(dataframe, min_comment_count, on='uid')

        min_like_count = grouped[['like_count']].min()
        min_like_count['min_like_count'] = min_like_count['like_count']
        min_like_count.drop(['like_count'], axis=1, inplace=True)
        user_infos.append(min_like_count)
        # dataframe = pd.merge(dataframe, min_like_count, on='uid')

        # mean feature
        mean_forward_count = grouped[['forward_count']].mean()
        mean_forward_count['mean_forward_count'] = mean_forward_count['forward_count']
        mean_forward_count.drop(['forward_count'], axis=1, inplace=True)
        user_infos.append(mean_forward_count)
        # dataframe = pd.merge(dataframe, mean_forward_count, on='uid')

        mean_comment_count = grouped[['comment_count']].mean()
        mean_comment_count['mean_comment_count'] = mean_comment_count['comment_count']
        mean_comment_count.drop(['comment_count'], axis=1, inplace=True)
        user_infos.append(mean_comment_count)
        # dataframe = pd.merge(dataframe, mean_comment_count, on='uid')

        mean_like_count = grouped[['like_count']].mean()
        mean_like_count['mean_like_count'] = mean_like_count['like_count']
        mean_like_count.drop(['like_count'], axis=1, inplace=True)
        user_infos.append(mean_like_count)
        # dataframe = pd.merge(dataframe, mean_like_count, on='uid')
        return user_infos

    """content's feature"""

    def handle_content(self, dataframe):
        topic_pattern = re.compile(r'#.+#')
        reference_pattern = re.compile(r'【.+】')
        url_pattern = re.compile(r'[a-zA-z]+://[^\s]*')
        dataframe['if_topic'] = dataframe['content'].apply(lambda x: 1 if topic_pattern.findall(x) else 0)
        dataframe['if_aite'] = dataframe['content'].apply(lambda x: 1 if '@' in x else 0)
        dataframe['if_reference'] = dataframe['content'].apply(lambda x: 1 if reference_pattern.findall(x) else 0)
        dataframe['if_url'] = dataframe['content'].apply(lambda x: 1 if url_pattern.findall(x) else 0)
        dataframe['char_lens'] = dataframe['content'].apply(lambda x: len(x))
        dataframe['if_keywords'] = dataframe['content'].apply(lambda x: self.assert_keyword(x))
        dataframe['count_keywords'] = dataframe['content'].apply(lambda x: self.count_keyword(x))
        dataframe.drop(['content'], axis=1, inplace=True)

    def assert_keyword(self, content):
        for keyword in self.keywords:
            if keyword in content:
                return 1
            else:
                return 0

    def count_keyword(self, content):
        count = 0
        for keyword in self.keywords:
            if keyword in content:
                count += 1
        return count

    def get_topk_word(self, dataframe):
        dataframe['content_str'] = dataframe['content'].apply(lambda x: str(x))
        content_list = dataframe['content_str'].to_list()
        content_list = [re.sub(r'[^\u4e00-\u9fa5]', '', content) for content in content_list]
        all_content_str = ''.join(content for content in content_list)
        keywords = self.tfidf(all_content_str)
        return keywords


m = Feature()
