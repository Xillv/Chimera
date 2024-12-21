import json
from sklearn.metrics import f1_score


def read_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def write_json(file_name, data, mode='w'):
    with open(file_name, mode) as f:
        json.dump(data, f)
        

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }

def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)






