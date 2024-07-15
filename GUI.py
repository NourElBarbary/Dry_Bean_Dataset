import tkinter as tk
from tkinter import ttk
import seaborn as sns
# from sklearn.linear_model import Perceptron, SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
import Perceptron
import Adaline
from  Preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from itertools import combinations
import warnings
 
from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
warnings.filterwarnings("ignore")



class DryBeansClassificationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dry Beans Classification")
        self.geometry("1500x800")

        # Features selection
        features_frame = ttk.LabelFrame(self, text="Select Features", padding=(10, 5))
        features_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.feature_vars = []
        self.feature_checkbuttons = []

        features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
        for i, feature in enumerate(features):
            var = tk.BooleanVar(value=False)
            self.feature_vars.append(var)
            checkbutton = ttk.Checkbutton(features_frame, text=feature, variable=var, style="TCheckbutton")
            checkbutton.grid(row=i, column=0, sticky="w", padx=(10, 0), pady=5)
            self.feature_checkbuttons.append(checkbutton)

        # Classes selection
        classes_frame = ttk.LabelFrame(self, text="Select Classes", padding=(10, 5))
        classes_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        classes_label = ttk.Label(classes_frame, text="Select classes:")
        classes_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")

        self.available_classes = ["BOMBAY", "CALI", "SIRA"]
        class_combinations = [' & '.join(comb) for comb in combinations(self.available_classes, 2)]
        self.selected_classes = tk.StringVar()  # Variable to store selected class combination
        self.classes_combobox = ttk.Combobox(classes_frame, values=class_combinations, textvariable=self.selected_classes)
        self.classes_combobox.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Learning parameters
        parameters_frame = ttk.LabelFrame(self, text="Learning Parameters", padding=(10, 5))
        parameters_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        lp_label = ttk.Label(parameters_frame, text="Enter learning rate:")
        lp_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        self.lp_entry = ttk.Entry(parameters_frame)
        self.lp_entry.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")

        epochs_label = ttk.Label(parameters_frame, text="Enter number of epochs:")
        epochs_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(parameters_frame)
        self.epochs_entry.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="ew")

        mse_label = ttk.Label(parameters_frame, text="Enter MSE threshold:")
        mse_label.grid(row=2, column=0, padx=(10, 5), pady=5, sticky="w")
        self.mse_entry = ttk.Entry(parameters_frame)
        self.mse_entry.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Bias checkbox
        self.bias_var = tk.BooleanVar()
        bias_checkbox = ttk.Checkbutton(self, text="Add Bias", variable=self.bias_var, style="TCheckbutton")
        bias_checkbox.grid(row=3, column=0, padx=10, pady=(10, 5), sticky="w")

        # Algorithm selection
        algorithm_frame = ttk.LabelFrame(self, text="Algorithm Selection", padding=(10, 5))
        algorithm_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.algorithm_var = tk.StringVar(value="Perceptron")
        perceptron_radio = ttk.Radiobutton(algorithm_frame, text="Perceptron", variable=self.algorithm_var, value="Perceptron", style="TRadiobutton")
        perceptron_radio.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        adaline_radio = ttk.Radiobutton(algorithm_frame, text="Adaline", variable=self.algorithm_var, value="Adaline", style="TRadiobutton")
        adaline_radio.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Classify button
        classify_button = ttk.Button(self, text="Classify", command=self.classify)
        classify_button.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        dashboard_frame = ttk.LabelFrame(self, text="Dashboard", padding=(10, 5))
        dashboard_frame.grid(row=0, column=1, rowspan=6, padx=10, pady=10, sticky="nsew")

        # Canvas for displaying graphs
        self.dashboard_canvas = FigureCanvasTkAgg(Figure(figsize=(10, 8)), master=dashboard_frame)
        self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def classify(self):
    # Fetch user inputs
        selected_features = [self.feature_checkbuttons[i]['text'] for i in range(len(self.feature_checkbuttons)) if self.feature_vars[i].get()]
        selected_classes_str = self.selected_classes.get()
        selected_classes = selected_classes_str.split(" & ")
        learning_rate = float(self.lp_entry.get())
        epochs = int(self.epochs_entry.get())
        mse_thresh = float(self.mse_entry.get())
        add_bias = self.bias_var.get()
        selected_algorithm = self.algorithm_var.get()
        # print("Selected Features:", selected_features)
        # print("Selected Classes:", selected_classes)
        # print("Learning Rate:", learning_rate)
        # print("Number of Epochs:", epochs)
        # print("MSE Threshold:", mse_thresh)
        # print("Add Bias:", add_bias)
        # print("Selected Algorithm:", selected_algorithm)
        # Read the data
        df = pd.read_csv("Dry_Bean_Dataset.csv")

        # Preprocess data
        X, y = preprocess_data(df, selected_features, selected_classes)



        # Train and test the classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=41)
        print(len(y_train))
        print(len(y_test))

        if selected_algorithm == "Perceptron":
            model = Perceptron.SLP(random_state=42,epochs=epochs,lr=learning_rate,bias=add_bias)
            model.fit(X_train, y_train)
            # print(model.weight)
            y_pred=model.predict(X_test)
            print("Predictions: ",y_pred)
            marker_shapes = {1: 'o', -1: 's'}
            # conf=model.confusion_matrix(y_true=y_test,y_pred=y_pred.flatten())
            # model.scatter_line(X_test,y_test)
            model.dashboard(self.dashboard_canvas,X_test,y_test,y_pred)

        else:  # Adaline
            model = Adaline.Adaline(random_state=42,epochs=epochs,lr=learning_rate,bias=add_bias)
            model.fit(X_train, y_train,mse_thresh)
            # print(model.weight)
            y_pred=model.predict(X_test)
            print("Predictions: ",y_pred)
            # conf=model.confusion_matrix(y_true=y_test,y_pred=y_pred.flatten())
            # model.scatter_line(X_test,y_test)
            model.dashboard(self.dashboard_canvas,X_test,y_test,y_pred.flatten())

if __name__=='__main__':
    app = DryBeansClassificationGUI()
    app.mainloop()











