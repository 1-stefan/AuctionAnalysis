import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from models.train_model_regression import LR_reg, KNN_reg, RandomForest_reg, SVM_reg
from models.train_model_classification import KNN_class, RandomForest_class,SVM_class

class viz:
    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
    def viz_reg(self):
        '''
        Creates graph of actual vs predicted values for regression problem
        '''
        ax1 = sns.distplot(self.y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(self.y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)
        plt.title('Actual vs Predicted Values for Price')
        #plt.xlabel('Price (in dollars)')
        #plt.ylabel('Proportion of Cars')
        plt.show()
        plt.close()
    def viz_class(self):
        '''
        Creates graph of confusion matrix for classification problem
        '''
        cm = confusion_matrix(self.y_pred, self.y_test)
        cm_df = pd.DataFrame(cm,
                             index = ['Cartier', 'Xbox'],
                             columns = ['Cartier', 'Xbox'])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_df)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()  

    
