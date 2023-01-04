import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def experiment(X, y, clf, verbose=0):
    test_y_total = []
    pred_y_total = []
    sensitivity_total = []
    specificity_total = []
    precision_total = []
    accuracy_total = []
    f1_total = []
    n_splits = 10

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)  # keeps the ratio of the classes in each fold

    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(X, y):
        # select rows
        train_X, test_X = X.to_numpy()[train_ix], X.to_numpy()[test_ix]
        train_y, test_y = y.to_numpy().reshape((-1,))[train_ix], y.to_numpy().reshape((-1,))[test_ix]

        # summarize train and test composition
        train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
        test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])



        clf.fit(train_X, train_y)


        pred_y = clf.predict(test_X)

        cf_matrix = metrics.confusion_matrix(test_y, pred_y, labels=[0, 1])
        print('test_y')
        print(test_y)
        print('pred_y')
        print(pred_y)
        print("Confusion matrix")
        print(cf_matrix)
        TP = cf_matrix[1][1]
        TN = cf_matrix[0][0]
        FP = cf_matrix[0][1]
        FN = cf_matrix[1][0]

        # for confusion matrix
        test_y_total = np.append(test_y_total, test_y)
        pred_y_total = np.append(pred_y_total, pred_y)

        # Specificity, Sensitivity, Accuracy and F1-measure.
        # Sensitivity = Recall = (True Positive)/(True Positive + False Negative)
        sensitivity = TP / (TP + FN)
        sensitivity_total += [sensitivity]
        print("sensitivity", sensitivity, TP, FN)

        # precision = tp / p = tp / (tp + fp)
        precision = TP / (TP + FP)
        precision_total += [precision]

        # accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy_total += [accuracy]

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = precision * sensitivity * 2 / (sensitivity + precision)
        if (precision == 0 and sensitivity == 0):
            f1 = 0
        f1_total += [f1]


        if verbose:
            print('>Train: 0=%d, 1=%d \t Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

        '''
    print(classification_report(test_y, pred_y, labels=[0,1]))
    print('Sensitivity: {0:0.2f}'.format(sensitivity))
    print('Precision:   {0:0.2f}'.format(precision))
    print('Specificity: {0:0.2f}'.format(specificity))
    print('Accuracy:    {0:0.2f}'.format(accuracy))
    print('f1:          {0:0.2f}'.format(f1))
    '''

    print('')
    report = metrics.classification_report(test_y_total, pred_y_total, labels=[0, 1])
    print(report)

    sen = np.array(sensitivity_total).mean()
    print('Sensitivity: {0:0.2f}'.format(sen))

    print('Precision:   {0:0.2f}'.format(np.array(precision_total).mean()))

    '''
    spe = np.array(specificity_total).mean()
    print('Specificity: {0:0.2f}'.format(spe))'''

    acc = np.array(accuracy_total).mean()
    print('Accuracy:    {0:0.2f}'.format(acc))

    f = np.array(f1_total).mean()
    print('f1:          {0:0.2f}'.format(f))

    cf_matrix_total = metrics.confusion_matrix(test_y_total, pred_y_total)
    matrix = showConfusionMatrix(cf_matrix_total, 'matrix.png')

def showConfusionMatrix(cf_matrix, save_path, type_arg="A", ):
    if type_arg == 'A':
        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf_matrix, annot=labels, fmt='', linewidths=1, cmap='Blues')
        plt.savefig(save_path)
        plt.show()
    elif type_arg == 'S':
        sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g')
        plt.savefig(save_path)
        plt.show()

    else:
        raise ValueError("Type argument in CFMatrix can be either 'S' or 'A'")
    return plt