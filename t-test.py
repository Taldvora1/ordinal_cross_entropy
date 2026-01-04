import numpy as np
_ordinal_cross_entropy_AUC = [0.767748727,
0.837694766,
0.868448754,
0.85560166,
0.865403694
]

_cross_entropy_AUC = [0.861258636,
0.774867431,
0.871645868,
0.835985981,
0.809891189]

_ordinal_cross_entropy_accuracy = [0.769230769,
0.745901639,
0.750819672,
0.762295082,
0.763934426

]

_cross_entropy_accuracy = [0.761047463,
0.750819672,
0.737704918,
0.740983607,
0.759016393
]

_ordinal_cross_entropy_cost = [1.145662848,
1.26557377,
1.281967213,
1.22295082,
1.203278689


]

_cross_entropy_cost = [1.191489362,
1.245901639,
1.370491803,
1.301639344,
1.259016393



]


def t_test(auc1, auc2):
    from scipy import stats
    import numpy as np

    auc1 = np.array(auc1)
    auc2 = np.array(auc2)

    t_statistic, p_value = stats.ttest_rel(auc1, auc2)
    return t_statistic, p_value

if __name__ == "__main__":
    t_statistic_AUC, p_value_AUC = t_test(_ordinal_cross_entropy_AUC, _cross_entropy_AUC)
    print(f"T-statistic: {t_statistic_AUC}, P-value: {p_value_AUC}")

    if p_value_AUC < 0.1:
        print("The difference between the two AUCs is statistically significant.")
    else:
        print("The difference between the two AUCs is not statistically significant.")
    t_statistic_accuracy, p_value_accuracy = t_test(_ordinal_cross_entropy_accuracy, _cross_entropy_accuracy)
    print(f"T-statistic: {t_statistic_accuracy}, P-value: {p_value_accuracy}")
    if p_value_accuracy < 0.1:
        print("The difference between the two accuracies is statistically significant.")
    else:
        print("The difference between the two accuracies is not statistically significant.")
    t_statistic_cost, p_value_cost = t_test(_ordinal_cross_entropy_cost, _cross_entropy_cost)
    print(f"T-statistic: {t_statistic_cost}, P-value: {p_value_cost}")
    if p_value_cost < 0.1:
        print("The difference between the two costs is statistically significant.")
    else:
        print("The difference between the two costs is not statistically significant.")
    print("Ordinal AUC mean:", np.mean(_ordinal_cross_entropy_AUC), "±", np.std(_ordinal_cross_entropy_AUC, ddof=1))
    print("Cross-Entropy AUC mean:", np.mean(_cross_entropy_AUC), "±", np.std(_cross_entropy_AUC, ddof=1))
    print("Ordinal Accuracy mean:", np.mean(_ordinal_cross_entropy_accuracy), "±", np.std(_ordinal_cross_entropy_accuracy, ddof=1))
    print("Cross-Entropy Accuracy mean:", np.mean(_cross_entropy_accuracy), "±", np.std(_cross_entropy_accuracy, ddof=1))
    print("Ordinal Cost mean:", np.mean(_ordinal_cross_entropy_cost), "±", np.std(_ordinal_cross_entropy_cost, ddof=1))
    print("Cross-Entropy Cost mean:", np.mean(_cross_entropy_cost), "±", np.std(_cross_entropy_cost, ddof=1))