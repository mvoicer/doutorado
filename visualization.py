from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class Visualization:

    @staticmethod
    def scatter_predicted(n_obj, y_test, y_pred):
        for j in range(n_obj):
            print("Real vs Predicted OBJECTIVE: ", j)
            plt.scatter(x=y_test.iloc[:, j], y=y_pred.iloc[:, j], c='crimson')
            p1 = max(max(y_pred.iloc[:, j]), max(y_test.iloc[:, j]))
            p2 = min(min(y_pred.iloc[:, j]), min(y_test.iloc[:, j]))
            plt.plot([p1, p2], [p1, p2], 'b-')
            plt.xlabel('True Values', fontsize=12)
            plt.ylabel('Predictions', fontsize=12)
            plt.ylim([-10, 10])
            plt.xlim([-10, 10])
            plt.title('Real vs Predict')
            plt.axis('equal')
            plt.show()

    @staticmethod
    def hist_residuals(n_obj, y_test, y_pred):
        for j in range(n_obj):
            print("Std residuals OBJECTIVE: ", j)
            sns.displot(y_test.iloc[:, j] - y_pred.iloc[:, j], kde=True, bins=50)
            plt.xlabel('Real - Predicted', fontsize=12)
            plt.ylabel('Standardized residuals', fontsize=12)
            plt.xlim([-10, 10])
            plt.title('Residuals')
            plt.show()

    @staticmethod
    def scatter_residuals(n_obj, y_test, y_pred):
        for j in range(n_obj):
            print("Residuals OBJECTIVE: ", j)
            sns.scatterplot(x=y_test.iloc[:, j], y=y_test.iloc[:, j] - y_pred.iloc[:, j])
            plt.xlabel('Real preferences')
            plt.ylabel('Residuals of the predicted preferences')
            plt.title('Residuals')
            plt.show()

    @staticmethod
    def plot_recommended_solutions(df_obj, recommended, ranking, rank_mcdm, n_rec):
        plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o', facecolors='none')
        plt.scatter(df_obj.iloc[recommended[-n_rec:], 0], df_obj.iloc[recommended[-n_rec:], 1],
                    color='blue', marker='o', label='Recommended')
        plt.scatter(df_obj.iloc[ranking[:n_rec], 0], df_obj.iloc[ranking[:n_rec], 1],
                    color='red', marker='o', label='Best predicted')
        plt.scatter(df_obj.iloc[rank_mcdm[:n_rec], 0], df_obj.iloc[rank_mcdm[:n_rec], 1],
                    color='black', marker='+', s=100, label='Best ahp')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.title('Recommended solutions')
        plt.legend(loc='best')
        plt.ylim([-0.05, 2.2])
        plt.xlim([-0.05, 2.2])
        plt.show()

    @staticmethod
    def plot_pareto_front(df_obj, rank_mcdm, n_rec):
        if df_obj.shape[1] == 2:
            plt.scatter(df_obj.iloc[:, 0],
                        df_obj.iloc[:, 1],
                        color='grey',
                        facecolors='none',
                        marker='o',
                        label='Available')
            plt.scatter(df_obj.iloc[rank_mcdm[:n_rec], 0],
                        df_obj.iloc[rank_mcdm[:n_rec], 1],
                        color='black', marker='+', s=100, label='Best ahp')
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.legend(loc='best')
            plt.title('Best solutions in Pareto front')
            plt.show()

        elif df_obj.shape[1] == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(df_obj.iloc[:, 0], df_obj.iloc[:, 1], df_obj.iloc[:, 2],
                         color='grey', marker='o', label='Available')
            ax.scatter3D(df_obj.iloc[rank_mcdm[:n_rec], 0], df_obj.iloc[rank_mcdm[:n_rec], 1],
                         df_obj.iloc[rank_mcdm[:n_rec], 2], color='black', marker='+', s=100, label='Best ahp')
            ax.view_init(30, 30)
            ax.set_xlabel('Obj 1')
            ax.set_ylabel('Obj 2')
            ax.set_zlabel('Obj 3')
            ax.legend(loc='best')
            ax.set_title('Best solutions in Pareto front')
            plt.show()

        else:
            # plot in parallel coordinates
            import plotly.express as px

            fig = px.parallel_coordinates(df_obj,
                                          dimensions=df_obj.columns,
                                          labels={k: v for k, v in
                                                  enumerate(['Obj ' + str(i) for i in df_obj.columns])})
            fig.show()
