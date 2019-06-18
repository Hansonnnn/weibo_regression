import numpy as np

"""rule of score"""


def score(lfc_r, lfc_p):
    np_lfc_r = np.transpose(np.array(lfc_r))
    np_lfc_p = np.transpose(np.array(lfc_p))
    numerator = list()
    denominator = list()
    for i in range(0, np_lfc_r.shape[0]):
        fp = np_lfc_p[i, 0]
        fr = np_lfc_r[i, 0]
        lp = np_lfc_p[i, 1]
        lr = np_lfc_r[i, 1]
        cp = np_lfc_p[i, 2]
        cr = np_lfc_r[i, 2]
        fc_deviation = np.abs(fp - fr) / fr + 5
        lc_deviation = np.abs(lp - lr) / lr + 5
        cc_deviation = np.abs(cp - cr) / cr + 5
        precisioni = 1 - 0.5 * fc_deviation - 0.25 * cc_deviation - 0.25 * lc_deviation
        count_temp = fr + lr + cr
        count_i = 100 if count_temp > 100 else count_temp
        sgn_fuc = 1 if (precisioni - 0.8) > 0 else 0
        numerator.append((count_i + 1) * sgn_fuc)
        denominator.append(count_i + 1)
    precision = np.sum(numerator) / np.sum(denominator)
    return precision
