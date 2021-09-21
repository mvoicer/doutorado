from matplotlib import pyplot as plt
import seaborn as sns


class Visualization:

    @staticmethod
    def scatter_predicted(n_obj, y_test, y_pred):
        for j in range(n_obj):
            sns.scatterplot(x = y_test.iloc[:, j], y = y_pred.iloc[:, j])
            plt.xlabel('Real preferences')
            plt.ylabel('Predicted through ML')
            plt.title('Real vs Predict values')
        plt.show()

    @staticmethod
    def hist_residuals(n_obj, y_test, y_pred):
        for j in range(n_obj):
            sns.distplot(y_test.iloc[:, j] - y_pred.iloc[:, j], bins = 50)
            plt.xlabel('Real - Predicted values')
            plt.ylabel('Standardized residuals')
            plt.title('Residuals')
        plt.show()

    @staticmethod
    def scatter_residuals(n_obj, y_test, y_pred):
        for j in range(n_obj):
            sns.scatterplot(x = y_test.iloc[:, obj], y = y_test.iloc[:, j] - y_pred.iloc[:, j])
            plt.xlabel('Real preferences')
            plt.ylabel('Residuals of the predicted preferences')
            plt.title('Residuals')
        plt.show()