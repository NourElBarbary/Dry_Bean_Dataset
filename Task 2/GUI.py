import tkinter as tk
from tkinter import ttk
import seaborn as sns

# from sklearn.linear_model import Perceptron, SGDRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Viz as v
from  Preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from itertools import combinations
import warnings
import Network
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

        # Learning parameters
        parameters_frame = ttk.LabelFrame(self, text="Learning Parameters", padding=(10, 5))
        parameters_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        hidden_layers_label = ttk.Label(parameters_frame, text="Enter number of hidden layers:")
        hidden_layers_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        self.hidden_layers_entry = ttk.Entry(parameters_frame)
        self.hidden_layers_entry.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")

        neurons_label = ttk.Label(parameters_frame, text="Enter number of neurons in each hidden layer:")
        neurons_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")
        self.neurons_entry = ttk.Entry(parameters_frame)
        self.neurons_entry.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="ew")

        lp_label = ttk.Label(parameters_frame, text="Enter learning rate (eta):")
        lp_label.grid(row=2, column=0, padx=(10, 5), pady=5, sticky="w")
        self.lp_entry = ttk.Entry(parameters_frame)
        self.lp_entry.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="ew")

        epochs_label = ttk.Label(parameters_frame, text="Enter number of epochs (m):")
        epochs_label.grid(row=3, column=0, padx=(10, 5), pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(parameters_frame)
        self.epochs_entry.grid(row=3, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Bias checkbox
        self.bias_var = tk.BooleanVar()
        bias_checkbox = ttk.Checkbutton(self, text="Add Bias", variable=self.bias_var, style="TCheckbutton")
        bias_checkbox.grid(row=2, column=0, padx=10, pady=(10, 5), sticky="w")

        # Activation function selection
        activation_frame = ttk.LabelFrame(self, text="Activation Function Selection", padding=(10, 5))
        activation_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.activation_var = tk.StringVar(value="Sigmoid")
        sigmoid_radio = ttk.Radiobutton(activation_frame, text="Sigmoid", variable=self.activation_var, value="sigmoid", style="TRadiobutton")
        sigmoid_radio.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        tanh_radio = ttk.Radiobutton(activation_frame, text="Hyperbolic Tangent Sigmoid", variable=self.activation_var, value="tanh", style="TRadiobutton")
        tanh_radio.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Classify button
        classify_button = ttk.Button(self, text="Classify", command=self.classify)
        classify_button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
    
        dashboard_frame = ttk.LabelFrame(self, text="Dashboard", padding=(10, 5))
        dashboard_frame.grid(row=0, column=1, rowspan=6, padx=10, pady=10, sticky="nsew")

        # Canvas for displaying graphs
        self.dashboard_canvas = FigureCanvasTkAgg(Figure(figsize=(10, 8)), master=dashboard_frame)
        self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        
        
    def classify(self):
        # Fetch user inputs
        selected_features = [self.feature_checkbuttons[i]['text'] for i in range(len(self.feature_checkbuttons)) if self.feature_vars[i].get()]
        activation = self.activation_var.get()
        learning_rate = float(self.lp_entry.get())
        epochs = int(self.epochs_entry.get())
        bias = self.bias_var.get()

        #  neural network
        hidden_layers = [int(layer) for layer in self.hidden_layers_entry.get()]

        
        df = pd.read_csv("Dry_Bean_Dataset.csv")

        
        X, y = preprocess_data(df,selected_features, ['BOMBAY', 'CALI',"SIRA"],activation)

        
        one_hot_encoded = pd.get_dummies(y)
        X_train, X_test, y_train, y_test = train_test_split(X, one_hot_encoded, test_size=0.2, random_state=42,stratify=y)
        network = Network.Network(len(hidden_layers), hidden_layers,42, learning_rate,activation , bias ,epochs)

        network.Train(X_train.to_numpy(), y_train.to_numpy())
        y_pred = network.Test(X_test)
        v.confusion_matrix(y_pred=y_pred,y_true= np.argmax(y_test.to_numpy(),axis=1),show=True)

        print("Accuracy", (np.argmax(y_test.to_numpy(),axis=1)==y_pred).sum()/y_test.shape[0])

if __name__=='__main__':
    app = DryBeansClassificationGUI()
    app.mainloop()








