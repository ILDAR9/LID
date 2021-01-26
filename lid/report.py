import os
from importlib import reload

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from lid import datautil, common
from lid.demoutils import LID


class Report:
    def __init__(self, setting_path, user_settings, df):
        self.predictor = LID(setting_path, user_settings)
        self.df = df

    def voting_mean(self):
        vot_method = self.predictor.settings['voting']
        self.predictor.settings['voting'] = 'mean'
        self._report_voting()
        self.predictor.settings['voting'] = vot_method

    def voting_majority(self):
        vot_method = self.predictor.settings['voting']
        self.predictor.settings['voting'] = 'majority'
        self._report_voting()
        self.predictor.settings['voting'] = vot_method

    def _report_voting(self):
        conf_mat = self.predictor.evaluate(self.df)
        ConfusionMatrixDisplay(conf_mat, display_labels=common.LABELS).plot()
        plt.title(f"Voting metrics: method is '{self.predictor.settings['voting']}'")
        plt.show()

    def train_process_plot(self):
        train_history_path = os.path.join(os.path.dirname(self.predictor.best_model_fname), 'train.csv')
        df_train = pd.read_csv(train_history_path)
        print(f"Loaded training metrics: {df_train.columns}")
        self.train_valmetric_plot(df_train)
        self.accuracy_vs_voted_plot(df_train)
        self.overfit_plot(df_train)

    @staticmethod
    def train_valmetric_plot(df):
        df.plot(y=['val_accuracy', 'voted_majority_acc', 'voted_mean_acc'], x='epoch', figsize=(12, 8))
        plt.title('Validation metrics while training process')
        plt.show()

    def accuracy_vs_voted_plot(self, df):
        df.plot.scatter(x='val_accuracy', y=f"voted_{self.predictor.settings['voting']}_acc", figsize=(8, 8))
        plt.title(f"Validation accuracy vs Voted-{self.predictor.settings['voting']} accuracy")
        plt.show()

    @staticmethod
    def overfit_plot(df):
        plt.plot(df.epoch, df.val_loss - df.loss)
        plt.title("Overfitting level")
        plt.xlabel('Epoch')
        plt.ylabel('loss diff')
        plt.show()
