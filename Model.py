import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd

class Model:

    def __init__(self,random_state=42,epochs=1000,lr=0.001,bias=True) -> None:
        self.weight=None
        self.pred=None
        self.learning_rate=lr
        self.epochs=epochs
        self.bias=bias


    def confusion_matrix(self,y_true,y_pred,show=True):
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
    
    def classification_report(self, y_true, y_pred):
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        num_labels = len(unique_labels)
        confusion_matrix = self.confusion_matrix(y_true,y_pred,False)


        # Compute precision, recall, and F1-score for each class
        precision = np.zeros(num_labels)
        recall = np.zeros(num_labels)
        f1 = np.zeros(num_labels)
        support = np.sum(confusion_matrix, axis=1)

        for i in range(num_labels):
            true_positive = confusion_matrix[i, i]
            false_positive = np.sum(confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(confusion_matrix[i, :]) - true_positive

            precision[i] = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall[i] = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0

        # Compute overall accuracy
        correct_predictions = np.sum(np.diag(confusion_matrix))
        total_predictions = np.sum(confusion_matrix)
        accuracy = correct_predictions / total_predictions

        # Format the output
        confusion_matrix_output = ""
        confusion_matrix_output += "   precision  recall  f1-score  support\n"
        for label, precision_value, recall_value, f1_value, support_value in zip(unique_labels, precision, recall, f1, support):
            confusion_matrix_output += f"{label}\t{precision_value:.2f}\t{recall_value:.2f}\t{f1_value:.2f}\t{support_value}\n"
        confusion_matrix_output += f"\naccuracy\t\t\t{accuracy:.2f}\n"

        return confusion_matrix_output
    
    def scatter_line(self,X_test,y_test):

        df=pd.concat([X_test,y_test],axis=1)
        df.columns=['f1','f2','l']
        sns.scatterplot(data=df, x='f1', y='f2',hue='l',palette='viridis')
        x_values = np.linspace(-1.5, 1, 40)
        if self.bias:
            y_values = (-self.weight[1][0] * x_values - self.weight[0][0]) / self.weight[2][0]
        else:
            y_values = (-self.weight[0][0] * x_values ) / self.weight[1][0]
        sns.lineplot(x=x_values, y=y_values, color='red', linestyle='--', label='boundry')
        plt.title("Decision Boundry")
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        plt.show()
        

    def dashboard(self,canvas,X_test,y_test,y_pred):

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        # Add Classification Report
        report=self.classification_report(y_test,y_pred.flatten())
        axes[0].set_title("Classification Report")
        axes[0].text(0.5, 0.5, report.replace('\t',' '), ha='center', va='center', fontsize=14)
        axes[0].axis('off')  # Turn off axis for this subplot
        

        # Add line plot
        df=pd.concat([X_test,y_test],axis=1)
        df.columns=['f1','f2','l']
        sns.scatterplot(data=df, x='f1', y='f2',hue='l',palette='viridis',ax=axes[1])
        x_values = np.linspace(-1.5, 1, 40)
        if self.bias:
            y_values = (-self.weight[1][0] * x_values - self.weight[0][0]) / self.weight[2][0]
        else:
            y_values = (-self.weight[0][0] * x_values ) / self.weight[1][0]
        sns.lineplot(x=x_values, y=y_values, color='red', linestyle='--', label='boundry',ax=axes[1])
        axes[1].set_title("Decision Boundry")
        axes[1].set_xlabel(X_test.columns[0])
        axes[1].set_ylabel(X_test.columns[1])

        # Add confusion matrix
        confusion_matrix=self.confusion_matrix(y_true=y_test,y_pred=y_pred.flatten(),show=False)
        ax=sns.heatmap(confusion_matrix,annot=True,ax=axes[2])
        axes[2].set_title("Confusion Matrix")
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("Actual")
        plt.tight_layout()
        # Update the canvas with the new figure
        canvas.figure = fig
        canvas.draw()
        

