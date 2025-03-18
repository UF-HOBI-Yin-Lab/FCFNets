from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def sensitivity(y_true, y_pred):
    sensitivity = recall_score(y_true, y_pred)
    return sensitivity

def f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1

def auc(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)  # y_prob: predicted probabilities
    return auc

def accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def mcc(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc