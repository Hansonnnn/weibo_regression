import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from evaluate import score


class Model:
    def __init__(self):
        self.train_set = []
        self.test_set = []
        self.predict_feature = None
        self.load_data()

    def load_data(self):
        for i in range(0, 3):
            train_data = pd.read_csv('./input/train_set' + str(i) + '.csv')
            test_data = pd.read_csv('./input/test_set' + str(i) + '.csv')
            self.train_set.append(train_data)
            self.test_set.append(test_data)
        self.predict_feature = pd.read_csv('./input/predict.csv')
        self.predict_feature.fillna(0, inplace=True)
        self.predict_feature.replace(-np.inf, 0, inplace=True)
        self.predict_feature.replace(np.inf, 0, inplace=True)

    def train(self):
        predict_uid = self.predict_feature['uid'].get_values()
        predict_mid = self.predict_feature['mid'].get_values()
        self.predict_feature.drop(['uid', 'mid', 'count_keywords'], axis=1, inplace=True)
        predict_stack = []
        predict_result = dict()
        for i in range(0, len(self.train_set)):
            lc = self.train_set[i]['like_count']
            fc = self.train_set[i]['forward_count']
            cc = self.train_set[i]['comment_count']
            lr = self.test_set[i]['like_count']
            fr = self.test_set[i]['forward_count']
            cr = self.test_set[i]['comment_count']
            labels = [lc, fc, cc]
            predict_cs_res = []
            label_name = ['like_count', 'forward_count', 'comment_count']

            self.train_set[i].drop(['like_count', 'forward_count', 'comment_count', 'uid', 'mid', 'count_keywords'],
                                   axis=1, inplace=True)
            self.test_set[i].drop(['like_count', 'forward_count', 'comment_count', 'uid', 'mid', 'count_keywords'],
                                  axis=1,
                                  inplace=True)

            print("the {} batch is fitting".format(i))

            for j in range(0, len(labels)):
                lm = LinearRegression()
                lm.fit(self.train_set[i], labels[j])
                predict_cs = lm.predict(self.test_set[i])
                predict_cs_res.append(predict_cs)
                predict = lm.predict(self.predict_feature)
                predict_df_predictions = [int(item) for item in predict]
                predict_result[label_name[j]] = predict_df_predictions

            precision = score([lr, fr, cr], predict_cs_res)
            print("the {} batch with precision is :{}".format(i, precision))
            predict_result['uid'] = list(predict_uid)
            predict_result['mid'] = list(predict_mid)
            predict_stack.append(predict_result)

        predict_lc_stack = [item['like_count'] for item in predict_stack]
        predict_fc_stack = [item['forward_count'] for item in predict_stack]
        predict_cc_stack = [item['comment_count'] for item in predict_stack]
        predict_lc_final = np.mean(predict_lc_stack, axis=0, dtype=np.int64)
        predict_fc_final = np.mean(predict_fc_stack, axis=0, dtype=np.int64)
        predict_cc_final = np.mean(predict_cc_stack, axis=0, dtype=np.int64)
        predict_result['like_count'] = predict_lc_final
        predict_result['forward_count'] = predict_fc_final
        predict_result['comment_count'] = predict_cc_final
        result = pd.DataFrame(predict_result)
        result.to_csv('./output/result.csv', index=False)


m = Model()
m.train()
