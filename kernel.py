import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import re


class Model:
    def __init__(self):
        self.SEED = 2000
        self.train_data = None
        self.test_data = None
        self.load_data()
        self.feature_engineering()

    def load_data(self):
        train_col_name = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count',
                          'content']
        test_col_name = ['uid', 'mid', 'time', 'content']
        self.train_data = pd.read_table('./input/weibo_train_data.txt',
                                        names=train_col_name)
        self.test_data = pd.read_table('./input/weibo_predict_data.txt', names=test_col_name)

    def split_data_set(self):
        pass

    def feature_engineering(self):
        # for data_frame in combine:
        self.handle_user(self.train_data[0:3000])
        # self.handle_date(data_frame)
        # self.handle_content(data_frame)
        # data_frame.drop(['uid', 'time', 'content'], axis=1, inplace=True)
        print("feature engineering ending...")

    """time's feature"""

    def handle_date(self, dataframe):
        dataframe['time'] = dataframe['time'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
        dataframe['weekday'] = dataframe['time'].dt.weekday
        dataframe['hour_seg'] = dataframe['time'].dt.hour
        dataframe['isWeekend'] = dataframe['weekday'].apply(lambda x: 1 if x == 5 or x == 6 else 0)
        dr = pd.date_range(start='2015-02-01', end='2015-07-31')
        cal = calendar()
        holidays = cal.holidays(start=dr.min(), end=dr.max())
        dataframe['date'] = dataframe['time'].dt.date
        dataframe['Holiday'] = dataframe['date'].apply(lambda x: 1 if x in holidays else 0)
        dataframe.drop('date', axis=1, inplace=True)

    """user's feature"""

    def handle_user(self, dataframe):
        # max feature
        max_forward_count = dataframe.groupby(['uid'])[['forward_count']].max()
        max_forward_count['max_forward_count'] = max_forward_count['forward_count']
        max_forward_count.drop(['forward_count'], axis=1, inplace=True)

        max_comment_count = dataframe.groupby(['uid'])[['comment_count']].max()
        max_comment_count['max_comment_count'] = max_comment_count['comment_count']
        max_comment_count.drop(['comment_count'], axis=1, inplace=True)

        max_like_count = dataframe.groupby(['uid'])[['like_count']].max()
        max_like_count['max_like_count'] = max_like_count['like_count']
        max_like_count.drop(['like_count'], axis=1, inplace=True)

        # min feature
        min_forward_count = dataframe.groupby(['uid'])[['forward_count']].min()
        min_forward_count['min_forward_count'] = min_forward_count['forward_count']
        min_forward_count.drop(['forward_count'], axis=1, inplace=True)

        min_comment_count = dataframe.groupby(['uid'])[['comment_count']].min()
        min_comment_count['min_comment_count'] = min_comment_count['comment_count']
        min_comment_count.drop(['comment_count'], axis=1, inplace=True)

        min_like_count = dataframe.groupby(['uid'])[['like_count']].min()
        min_like_count['min_like_count'] = min_like_count['like_count']
        min_like_count.drop(['like_count'], axis=1, inplace=True)

        # mean feature
        mean_forward_count = dataframe.groupby(['uid'])[['forward_count']].mean()
        mean_forward_count['mean_forward_count'] = mean_forward_count['forward_count']
        mean_forward_count.drop(['forward_count'], axis=1, inplace=True)

        mean_comment_count = dataframe.groupby(['uid'])[['comment_count']].mean()
        mean_comment_count['mean_comment_count'] = mean_comment_count['comment_count']
        mean_comment_count.drop(['comment_count'], axis=1, inplace=True)

        mean_like_count = dataframe.groupby(['uid'])[['like_count']].mean()
        mean_like_count['mean_like_count'] = mean_like_count['like_count']
        mean_like_count.drop(['like_count'], axis=1, inplace=True)

        result = pd.concat(
            [max_comment_count, max_forward_count, max_like_count, min_comment_count, min_forward_count, min_like_count,
             mean_comment_count, mean_forward_count, mean_like_count], axis=1, join_axes=[max_comment_count.index],
        )
        result['uid'] = result.index
        # max comment/like, forward/like
        result['max_clr'] = result['max_comment_count'].div(result['max_like_count'])
        result['max_cfl'] = result['max_forward_count'].div(result['max_like_count'])
        # min comment/like, forward/like
        result['min_clr'] = result['min_comment_count'].div(result['min_like_count'])
        result['min_cfl'] = result['min_forward_count'].div(result['min_like_count'])
        # mean comment/like, forward/like
        result['mean_clr'] = result['mean_comment_count'].div(result['mean_like_count'])
        result['mean_cfl'] = result['mean_forward_count'].div(result['mean_like_count'])

        max_like_mean = dict()
        max_comment_mean = dict()
        max_forward_mean = dict()
        for user in result['uid']:
            u_info = dataframe[dataframe['uid'] == user]
            count1 = 0
            count2 = 0
            count3 = 0
            for i in u_info.index:
                f_info = result[result['uid'] == user]
                if u_info.loc[[i]]['like_count'].values[0] > f_info['mean_like_count'].values[0]:
                    count1 += 1

                if u_info.loc[[i]]['comment_count'].values[0] > f_info['mean_comment_count'].values[0]:
                    count2 += 1

                if u_info.loc[[i]]['forward_count'].values[0] > f_info['mean_forward_count'].values[0]:
                    count3 += 1
            mlm = count1 / u_info.ndim
            max_like_mean.update({user: [mlm]})

            mcm = count2 / u_info.ndim
            max_comment_mean.update({user: [mcm]})

            mfm = count3 / u_info.ndim
            max_forward_mean.update({user: [mfm]})

        max_like_mean_df = pd.DataFrame(data=max_like_mean.values(), index=max_like_mean.keys(),
                                        columns=['max_like_mean'])
        max_comment_mean_df = pd.DataFrame(data=max_comment_mean.values(), index=max_comment_mean.keys(),
                                           columns=['max_comment_mean'])
        max_forward_mean_df = pd.DataFrame(data=max_forward_mean.values(), index=max_forward_mean.keys(),
                                           columns=['max_forward_mean'])
        final_result = pd.concat([result, max_like_mean_df, max_comment_mean_df, max_forward_mean_df], axis=1,
                                 join_axes=[result.index])

        final_result.fillna(-1, axis=1)
        print(result.head())

    """content's feature"""

    def handle_content(self, dataframe):
        topic_pattern = re.compile('#.{1,}#')
        reference_pattern = re.compile('【.{1,}】')
        url_pattern = re.compile('[a-zA-z]+://[^\s]*')
        dataframe['if_topic'] = dataframe['content'].apply(lambda x: 1 if topic_pattern.findall(x) else 0)
        dataframe['if_aite'] = dataframe['content'].apply(lambda x: 1 if '@' in x else 0)
        dataframe['if_reference'] = dataframe['content'].apply(lambda x: 1 if reference_pattern.findall(x) else 0)
        dataframe['if_url'] = dataframe['content'].apply(lambda x: 1 if url_pattern.findall(x) else 0)
        dataframe['char_lens'] = dataframe['content'].apply(lambda x: len(x))

    def train(self):
        gb_params = {
            'n_estimators': 500,
            # 'max_features': 0.2,
            'max_depth': 10,
            'min_samples_leaf': 2,
            'verbose': 0
        }
        y_train1 = self.train_data['forward_count']
        y_train2 = self.train_data['comment_count']
        y_train3 = self.train_data['like_count']
        self.train_data.drop(['forward_count', 'comment_count', 'like_count'], axis=1, inplace=True)
        x_train = self.train_data
        x_test = self.test_data
        gbdt = GradientBoostingRegressor(gb_params=gb_params)

        print('start train...')

        kf = KFold(n_splits=10, shuffle=True, random_state=self.random_seed)
        predict_full_prob = 0
        predict_score = []
        count = 1
        for train_index, test_index in kf.split(x_train):
            print('{} of KFlod {}'.format(count, kf.n_splits))
            x1, x2 = x_train[train_index], x_train[test_index]
            y1, y2 = y_train1[train_index], y_train1[test_index]
            gbdt.fit(x1, y1)
            y_predict = gbdt.predict(x2)
            predict_full_prob += gbdt.predict_proba(x_test)
            count += 1

        submit = pd.DataFrame([self.test_data['uid'], self.test_data['mid']])
        submit = submit.join(pd.DataFrame(predict_full_prob))
        submit.columns = ['uid', 'mid', 'forward_count']
        submit.to_csv('../input/weibo_predict1.csv', index=False)


m = Model()
