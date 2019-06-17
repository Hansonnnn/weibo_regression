import numpy as np

"""rule of score"""


def score(lfc_r, lfc_p):
    np_lfc_r = np.array(lfc_r)
    np_lfc_p = np.array(lfc_p)
    numerator = list()
    denominator = list()
    for i in range(np_lfc_r.shape[0]):
        fp = np_lfc_p[0, i]
        fr = np_lfc_r[0, i]
        lp = np_lfc_p[1, i]
        lr = np_lfc_r[1, i]
        cp = np_lfc_p[2, i]
        cr = np_lfc_r[2, i]
        fc_deviation = np.abs(fp - fr) / fr + 5
        lc_deviation = np.abs(lp - lr) / lr + 5
        cc_deviation = np.abs(cp - cr) / cr + 5
        precision = 1 - 0.5 * fc_deviation - 0.25 * cc_deviation - 0.25 * lc_deviation
        count_temp = fr + lr + cr
        count_i = count_temp if count_temp > 100 else 100
        sgn_fuc = 1 if (precision - 0.8) > 0 else 0
        numerator.append((count_i + 1) * sgn_fuc)
        denominator.append(count_i + 1)
    precision = np.sum(numerator) / np.sum(denominator)
    return precision
