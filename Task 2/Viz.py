import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd

def confusion_matrix(y_true,y_pred,show=True):
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        num_labels = len(unique_labels)
        confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        for true_label, pred_label in zip(y_true, y_pred):
            true_index = label_to_index[true_label]
            pred_index = label_to_index[pred_label]
            confusion_matrix[true_index, pred_index] += 1
        if show:
            ax=sns.heatmap(confusion_matrix,annot=True)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
        return confusion_matrix
    