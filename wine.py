import warnings

import inline
import matplotlib
from sklearn import cluster
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB

warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
from matplotlib import pyplot as plt

from scipy import interp
from sklearn import metrics
from sklearn.preprocessing import label_binarize, StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)                  #para poder visualizar todas as colunas no display
pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', None)                  #para poder visualizar todas as colunas no display
pd.set_option('display.width', 1000)                        # para a largura do display ser de dimensao 1000

redwines = pd.read_csv("redwine.csv",delimiter=";")
whitewines = pd.read_csv("whitewine.csv",delimiter=";")
redwines.columns = redwines.columns.str.replace(' ', '_')       # torna mais facil a utilizaçao das colunas
whitewines.columns = whitewines.columns.str.replace(' ', '_')   # torna mais facil a utilizaçao das colunas

def menu():
    print("--------- Initial Menu ---------")
    print("[1] Business & Data Understanding")
    print("[2] Data Preparation")
    print("[3] Modeling (ML Algorithms Application) ")
    print("[4] Evaluation")
    print("[5] Deployment")

    print("\n[0] Exit the program")

menu()
option = int(input("\nInsert the command that you want to execute:\n"))


def submenu1():
    print("\n")
    print("------ Business & Data Understanding Menu ------")
    print("[1] Print to 50 first instances for each wine dataset")
    print("[2] Preparing data for the dataset")
    print("[3] Check how many null values exist")
    print("[4] Create descriptive statistics (only for numeric columns)")
    print("[5] Histograms showing all components of white and red wine")
    print("[6] Distribution of red and white wine taking into account the quality")
    print("[7] Boxplots showing all components of white and red wine")
    print("[8] Existing correlation between quality and the various attributes")
    print("[9] HeatMap - display correlation between red wine and white wine attributes")
    print("[10] Existing covariance between quality and the various attributes")
    print("[11] Red wine boxplots - characteristics for which it had the highest correlation")
    print("[12] White wine boxplots - characteristics for which it had the highest correlation")
    print("[13] Red Wine Regplots - express interesting correlations between different components")
    print("[14] White Wine Regplots - express interesting correlations between different components")
    print("[15] ScatterPlot matrix for red and white wines")

    print("\n[0] Return to the program's main menu")


def submenu2():
    print("\n")
    print("------ Data Preparation Menu ------")
    print("[1] Data Discretization")
    print("[2] Data Integration")
    print("[3] Data Cleaning")
    print("[4] Data Transformation - Creation of the Quality Label")
    print("[5] Data Transformation - Violin Plot with the Acidity")
    print("[6] Data Transformation - Removing Quality Not Discretized")

    print("\n[0] Return to the program's main menu")


def submenu3():
    print("\n")
    print("------ Modeling (ML Algorithms Application) Menu ------")
    print("[1] Balanced Data")
    print("[2] Not Balanced Data")

    print("\n[0] Return to the program's main menu")

def submenu4():
    print("\n")
    print("------ Modeling (ML Algorithms Application) Menu ------")
    print("[1] NaiveBayes")
    print("[2] KNN")
    print("[3] Logistic Regression")
    print("[4] Decision Tree")
    print("[5] Model Tree")
    print("[6] Artificial Neural Networks")
    print("[7] Support Vector Machine")
    print("[8] Random Forest")

    print("\n[0] Return to the program's main menu")

def submenu5():
    print("\n")
    print("------ Modeling (ML Algorithms Application) Menu ------")
    print("[1] NaiveBayes")
    print("[2] KNN")
    print("[3] Logistic Regression")
    print("[4] Decision Tree")
    print("[5] Model Tree")
    print("[6] Artificial Neural Networks")
    print("[7] Support Vector Machine")
    print("[8] Random Forest")

    print("\n[0] Return to the program's main menu")

    """
    print("[4] Cross-Validation and Training and Test Set - Red Wine")
    print("[5] Cross-Validation and Training and Test Set - White Wine")
    print("[8] Confusion Matrix")
    print("[9] Accuracy models - White Wine")
    print("[10] Accuracy models - Red Wine")
    print("[11] Tabela - White Wine")
    print("[12] Tabela - Red Wine")
    print("[13] K-Means Clustering - White Wine")
    print("[14] K-Means Clustering - Red Wine")
    print("[15] Hierarchical Clustering")
    print("[18] Time - White Wine")
    print("[19] Time - Red Wine")"""

    print("[0] Return to the program's main menu")


while option != 0:
    if option == 1:
        submenu1()
        option = int(input("\nInsert the command that you want to execute:\n"))

        while option != 0:
            if option == 1:
                print('\n')
                print('Red wine dataset')
                print(redwines.head(50))
                print('\n')
                print('White wine dataset')
                print(whitewines.head(50))

            elif option == 2:
                print('\n')
                print('Number of matrix elements for red wine')
                print(redwines.size)
                print('Number of matrix elements for white wine')
                print(whitewines.size)
                print('\n')
                print('Matrix dimension for red wines')
                print(redwines.shape)
                print('Matrix dimension for white wines')
                print(whitewines.shape)
                print('\n')
                print("Total number of data in each attribute for red wine")
                print(redwines.count())
                print("Total number of data in each attribute for white wine")
                print(whitewines.count())
                print('\n')
                print("Type of variables and amount of data in each column for red wine")
                print(redwines.info())
                print("Type of variables and amount of data in each column for white wine")
                print(whitewines.info())

            elif option == 3:
                nulos_red = redwines.isnull().sum()
                nulos_white = whitewines.isnull().sum()
                print('\n')
                print('Red wine dataset')
                print(nulos_red)
                print('\n')
                print('White wine dataset')
                print(nulos_white)
                print('\n')
                print("There are no null values in both datasets!!!")

            elif option == 4:
                print("\n")
                print("Red Wine")
                print(redwines.describe())
                print("\n")
                print("White Wine")
                print(whitewines.describe())
                print("\n")

            elif option == 5:
                redwines.hist()
                plt.tight_layout(pad=1.1)
                plt.suptitle('Red Wine - distribution of the various components', fontsize=13)
                whitewines.hist()
                plt.tight_layout(pad=1.1)
                plt.suptitle('White Wine - distribution of the various components', fontsize=13)
                plt.show()

            elif option == 6:
                sns.violinplot(x="quality", data=redwines)
                plt.suptitle('Distribution of Red Wine taking into account the quality', fontsize=12)
                plt.figure()
                sns.violinplot(x="quality", data=whitewines)
                plt.suptitle('Distribution of White Wine taking into account the quality', fontsize=12)
                plt.show()

            elif option == 7:
                b1 = sns.boxplot(data=redwines, palette='Blues')
                b1.set_xticklabels(b1.get_xticklabels(), rotation=90)
                plt.suptitle('Red Wine - distribution of the various components', fontsize=13)
                plt.tight_layout()
                plt.figure()
                b2 = sns.boxplot(data=whitewines, palette='Blues')
                b2.set_xticklabels(b2.get_xticklabels(), rotation=90)
                plt.suptitle('White Wine - distribution of the various components', fontsize=13)
                plt.tight_layout()
                plt.show()

            elif option == 8:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                correlationred = redwines.select_dtypes(include=numerics).corr()['quality'].sort_values(ascending=False)
                correlationwhite = whitewines.select_dtypes(include=numerics).corr()['quality'].sort_values(ascending=False)
                print("\n")
                print('Red Wine')
                print(correlationred)
                print("\n")
                print('White Wine')
                print(correlationwhite)

            elif option == 9:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                corr_red = redwines.select_dtypes(include=numerics).corr()
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr_red, xticklabels=corr_red.columns, yticklabels=corr_red.columns, annot=True,
                            cmap=sns.diverging_palette(220, 20, as_cmap=True))
                plt.suptitle('HeatMap- Correlation between red wine attributes', fontsize=15)
                plt.tight_layout()

                corr_white = whitewines.select_dtypes(include=numerics).corr()
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr_white, xticklabels=corr_white.columns, yticklabels=corr_white.columns, annot=True,
                            cmap=sns.diverging_palette(220, 20, as_cmap=True))
                plt.suptitle('HeatMap- Correlation between white wine attributes', fontsize=15)
                plt.tight_layout()

                plt.show()

            elif option == 10:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                covarianciared = redwines.select_dtypes(include=numerics).cov()['quality'].sort_values(ascending=False)
                covarianciawhite = whitewines.select_dtypes(include=numerics).cov()['quality'].sort_values(ascending=False)

                print("\n")
                print('Red Wine')
                print(covarianciared)
                print("\n")
                print('White Wine')
                print(covarianciawhite)

                corr = redwines.select_dtypes(include=numerics).cov()
                print('\n')
                print('Covariances between the various attributes of red wine')
                print(corr)
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                        cmap=sns.diverging_palette(220, 20, as_cmap=True))

                plt.suptitle('Covariances between the various attributes of red wine', fontsize=15)
                plt.tight_layout()

                corr1 = whitewines.select_dtypes(include=numerics).cov()
                print('\n')
                print('Covariances between the various attributes of white wine')
                print(corr1)
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr1, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                        cmap=sns.diverging_palette(220, 20, as_cmap=True))

                plt.suptitle('Covariances between the various attributes of white wine', fontsize=15)
                plt.tight_layout()
                plt.show()

            elif option == 11:
                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & alcohol', fontsize=15)
                sns.boxplot(orient='v', data=redwines, y="alcohol", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & sulfates', fontsize=15)
                sns.boxplot(orient='v', data=redwines, y="sulphates", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & volatile acidity ', fontsize=15)
                sns.boxplot(orient='v', data=redwines, y="volatile_acidity", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & citric acid', fontsize=15)
                sns.boxplot(orient='v', data=redwines, y="citric_acid", x="quality")

                plt.show()

            elif option == 12:

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & alcohol', fontsize=15)
                sns.boxplot(orient='v', data=whitewines, y="alcohol", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & sulfates', fontsize=15)
                sns.boxplot(orient='v', data=whitewines, y="sulphates", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & pH', fontsize=15)
                sns.boxplot(orient='v', data=whitewines, y="pH", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & density', fontsize=15)
                sns.boxplot(orient='v', data=whitewines, y="density", x="quality")

                plt.show()

            elif option == 13:
                fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
                sns.regplot(x="alcohol", y="density", data=redwines, ax=ax1, scatter_kws={'s': 2})
                sns.regplot(x="fixed_acidity", y="density", data=redwines, ax=ax2, scatter_kws={'s': 2})
                sns.regplot(x="citric_acid", y="fixed_acidity", data=redwines, ax=ax3, scatter_kws={'s': 2})
                sns.regplot(x="pH", y="fixed_acidity", data=redwines, ax=ax4, scatter_kws={'s': 2})
                plt.suptitle("Red Wine - Interesting correlations")
                plt.show()

            elif option == 14:
                fig, ((ax1, ax2) , (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
                sns.regplot(x="alcohol", y="density", data=whitewines,ax=ax1, scatter_kws={'s': 2})
                sns.regplot(x="total_sulfur_dioxide", y="density", data=whitewines, ax=ax2, scatter_kws={'s': 2})
                sns.regplot(x="density", y="residual_sugar", data=whitewines, ax=ax3, scatter_kws={'s': 2})
                sns.regplot(x="alcohol", y="residual_sugar", data=whitewines, ax=ax4, scatter_kws={'s': 2})
                plt.suptitle("White Wine - Interesting correlations")
                plt.show()

            elif option == 15:
                axes = pd.plotting.scatter_matrix(redwines, alpha=1.0)
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(25)
                    ax.yaxis.label.set_rotation(25)
                    ax.yaxis.label.set_ha('right')
                plt.suptitle('Red Wine - scatterplot of all attributes', fontsize=13)

                axes1 = pd.plotting.scatter_matrix(whitewines, alpha=1.0)
                for ax in axes1.flatten():
                    ax.xaxis.label.set_rotation(25)
                    ax.yaxis.label.set_rotation(25)
                    ax.yaxis.label.set_ha('right')
                plt.suptitle('White Wine - scatterplot of all attributes', fontsize=13)
                plt.show()

            else:
                print("Invalid option")

            print("\n")
            submenu1()
            option = int(input("\nInsert the command that you want to execute:"))
    elif option == 2:
        submenu2()
        option = int(input("\nInsert the command that you want to execute:\n"))

        while option != 0:
            if option == 1:
                redwines["wine_type"] = 0         #Red wine - 0
                whitewines["wine_type"] = 1       #White wine - 1
                print("Red Wine")
                print(redwines.head())
                print("\nWhite Wine")
                print(whitewines.head())

            elif option == 2:
                wines = pd.concat([redwines, whitewines], axis=0)
                wines.columns = wines.columns.str.replace(' ', '_')
                print("\n All wines together")
                print(wines)
                redwines = redwines.drop(columns=['wine_type'])
                whitewines = whitewines.drop(columns=['wine_type'])

            elif option == 3:
                wines = wines.drop_duplicates()
                wines = wines.dropna()                    # dropping the rows having NaN values
                wines = wines.reset_index(drop=True)      # To reset the indices
                print(wines)

            elif option == 4:

                wines_binary = wines.copy()
                wines_binary['quality_label'] = wines.quality.apply(
                    lambda q: 'low' if q <= 5 else 'medium' if q <= 7 else 'high')
                #last_column = wines_binary.pop('quality')
                #wines_binary.insert(12, 'quality', last_column)

                print(wines_binary)

                my_colors = ["#FF9999", "white"]
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                f.suptitle('Wine Type - Quality - Alcohol Content', fontsize=14)

                sns.boxplot(x='quality', y='alcohol', hue='wine_type', data=wines_binary, palette=my_colors, ax=ax1)
                ax1.set_xlabel("Wine Quality", size=12, alpha=0.8)
                ax1.set_ylabel("Wine Alcohol %", size=12, alpha=0.8)

                sns.boxplot(x='quality_label', y='alcohol', hue='wine_type', data=wines_binary, palette=my_colors, ax=ax2)
                ax2.set_xlabel("Wine Quality Class", size=12, alpha=0.8)
                ax2.set_ylabel("Wine Alcohol %", size=12, alpha=0.8)
                plt.show()

            elif option == 5:
                my_colors = ["#FF9999", "white"]
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                f.suptitle('Wine Type - Quality - Acidity', fontsize=14)

                sns.violinplot(x='quality', y='volatile_acidity', hue='wine_type', data=wines_binary, split=True, inner='quart',
                               linewidth=1.3,
                               palette=my_colors, ax=ax1)
                ax1.set_xlabel("Wine Quality", size=12, alpha=0.8)
                ax1.set_ylabel("Wine Fixed Acidity", size=12, alpha=0.8)

                sns.violinplot(x='quality_label', y='volatile_acidity', hue='wine_type', data=wines_binary, split=True,
                               inner='quart', linewidth=1.3,
                               palette=my_colors, ax=ax2)
                ax2.set_xlabel("Wine Quality Class", size=12, alpha=0.8)
                ax2.set_ylabel("Wine Fixed Acidity", size=12, alpha=0.8)
                plt.show()

            elif option == 6:
                wines_norm = wines.copy()

                for column in wines_norm[
                    ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                     'free_sulfur_dioxide',
                     'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']].columns:
                    wines_norm[column] = (wines_norm[column] -
                                          wines_norm[column].mean()) / wines_norm[column].std()

                # DISCRETIZE QUALITY TO NOT NORMALIZED DATASET
                wines_binary = wines.copy()
                wines_binary['quality'] = wines.quality.apply(
                    lambda q: 'low' if q <= 5 else 'medium' if q <= 7 else 'high')
                last_column = wines_binary.pop('quality')
                wines_binary.insert(12, 'quality', last_column)

                # DISCRETIZE QUALITY TO NORMALIZED DATASET
                wines_binary_norm = wines_norm.copy()
                wines_binary_norm['quality'] = wines.quality.apply(
                    lambda q: 'low' if q <= 4 else 'medium' if q <= 6 else 'high')
                last_column = wines_binary_norm.pop('quality')
                wines_binary_norm.insert(12, 'quality', last_column)
                print('\n')
                print('Normalized dataset ready')
                print(wines_binary_norm)
                print('\n')
                print('Not Normalized dataset ready')
                print(wines_binary)

                # visualizations before and after normalizing the data
                sns.displot(wines_norm['alcohol']).set(title='Normalized')
                plt.tight_layout()
                sns.displot(wines['alcohol']).set(title='Not Normalized')
                plt.tight_layout()
                plt.show()


                def get_results(model, name, data, true_labels, target_names=['red', 'white'], results=None,
                                reasume=False):

                    if hasattr(model, 'layers'):
                        param = wtp_dnn_model.history.params
                        best = np.mean(wtp_dnn_model.history.history['val_acc'])
                        predicted_labels = model.predict_classes(data)
                        im_model = InMemoryModel(model.predict, examples=data, target_names=target_names)

                    else:
                        param = gs.best_params_
                        best = gs.best_score_
                        predicted_labels = model.predict(data).ravel()
                        if hasattr(model, 'predict_proba'):
                            im_model = InMemoryModel(model.predict_proba, examples=data, target_names=target_names)
                        elif hasattr(clf, 'decision_function'):
                            im_model = InMemoryModel(model.decision_function, examples=data, target_names=target_names)

                    print('Mean Best Accuracy: {:2.2%}'.format(best))
                    print('-' * 60)
                    print('Best Parameters:')
                    print(param)
                    print('-' * 60)

                    y_pred = model.predict(data).ravel()

                    display_model_performance_metrics(true_labels, predicted_labels=predicted_labels,
                                                      target_names=target_names)
                    if len(target_names) == 2:
                        ras = roc_auc_score(y_true=true_labels, y_score=y_pred)
                    else:
                        roc_auc_multiclass, ras = roc_auc_score_multiclass(y_true=true_labels, y_score=y_pred,
                                                                           target_names=target_names)
                        print('\nROC AUC Score by Classes:\n', roc_auc_multiclass)
                        print('-' * 60)

                    print('\n\n              ROC AUC Score: {:2.2%}'.format(ras))
                    prob, score_roc, roc_auc = plot_model_roc_curve(model, data, true_labels, label_encoder=None,
                                                                    class_names=target_names)

                    interpreter = Interpretation(data, feature_names=cols)
                    plots = interpreter.feature_importance.plot_feature_importance(im_model, progressbar=False,
                                                                                   n_jobs=1, ascending=True)

                    r1 = pd.DataFrame([(prob, best, np.round(accuracy_score(true_labels, predicted_labels), 4),
                                        ras, roc_auc)], index=[name],
                                      columns=['Prob', 'CV Accuracy', 'Accuracy', 'ROC AUC Score', 'ROC Area'])
                    if reasume:
                        results = r1
                    elif (name in results.index):
                        results.loc[[name], :] = r1
                    else:
                        results = results.append(r1)

                    return results


                def roc_auc_score_multiclass(y_true, y_score, target_names, average="macro"):

                    # creating a set of all the unique classes using the actual class list
                    unique_class = set(y_true)
                    roc_auc_dict = {}
                    mean_roc_auc = 0
                    for per_class in unique_class:
                        # creating a list of all the classes except the current class
                        other_class = [x for x in unique_class if x != per_class]

                        # marking the current class as 1 and all other classes as 0
                        new_y_true = [0 if x in other_class else 1 for x in y_true]
                        new_y_score = [0 if x in other_class else 1 for x in y_score]
                        num_new_y_true = sum(new_y_true)

                        # using the sklearn metrics method to calculate the roc_auc_score
                        roc_auc = roc_auc_score(new_y_true, new_y_score, average=average)
                        roc_auc_dict[target_names[per_class]] = np.round(roc_auc, 4)
                        mean_roc_auc += num_new_y_true * np.round(roc_auc, 4)

                    mean_roc_auc = mean_roc_auc / len(y_true)
                    return roc_auc_dict, mean_roc_auc


                def get_metrics(true_labels, predicted_labels):

                    print('Accuracy:  {:2.2%} '.format(metrics.accuracy_score(true_labels, predicted_labels)))
                    print('Precision: {:2.2%} '.format(
                        metrics.precision_score(true_labels, predicted_labels, average='weighted')))
                    print('Recall:    {:2.2%} '.format(
                        metrics.recall_score(true_labels, predicted_labels, average='weighted')))
                    print('F1 Score:  {:2.2%} '.format(
                        metrics.f1_score(true_labels, predicted_labels, average='weighted')))


                def train_predict_model(classifier, train_features, train_labels, test_features, test_labels):
                    # build model
                    classifier.fit(train_features, train_labels)
                    # predict using model
                    predictions = classifier.predict(test_features)
                    return predictions


                def display_confusion_matrix(true_labels, predicted_labels, target_names):

                    total_classes = len(target_names)
                    level_labels = [total_classes * [0], list(range(total_classes))]

                    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
                    cm_frame = pd.DataFrame(data=cm,
                                            columns=pd.MultiIndex(levels=[['Predicted:'], target_names],
                                                                  codes=level_labels),
                                            index=pd.MultiIndex(levels=[['Actual:'], target_names], codes=level_labels))
                    print(cm_frame)


                def display_classification_report(true_labels, predicted_labels, target_names):

                    report = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels,
                                                           target_names=target_names)
                    print(report)


                def display_model_performance_metrics(true_labels, predicted_labels, target_names):
                    print('Model Performance metrics:')
                    print('-' * 30)
                    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
                    print('\nModel Classification report:')
                    print('-' * 30)
                    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels,
                                                  target_names=target_names)
                    print('\nPrediction Confusion Matrix:')
                    print('-' * 30)
                    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels,
                                             target_names=target_names)


                def plot_model_roc_curve(clf, features, true_labels, label_encoder=None, class_names=None):

                    ## Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    if hasattr(clf, 'classes_'):
                        class_labels = clf.classes_
                    elif label_encoder:
                        class_labels = label_encoder.classes_
                    elif class_names:
                        class_labels = class_names
                    else:
                        raise ValueError('Unable to derive prediction classes, please specify class_names!')
                    n_classes = len(class_labels)

                    if n_classes == 2:
                        if hasattr(clf, 'predict_proba'):
                            prb = clf.predict_proba(features)
                            if prb.shape[1] > 1:
                                y_score = prb[:, prb.shape[1] - 1]
                            else:
                                y_score = clf.predict(features).ravel()
                            prob = True
                        elif hasattr(clf, 'decision_function'):
                            y_score = clf.decision_function(features)
                            prob = False
                        else:
                            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")

                        fpr, tpr, _ = roc_curve(true_labels, y_score)
                        roc_auc = auc(fpr, tpr)

                        plt.plot(fpr, tpr, label='ROC curve (area = {0:3.2%})'.format(roc_auc), linewidth=2.5)

                    elif n_classes > 2:
                        if hasattr(clf, 'clfs_'):
                            y_labels = label_binarize(true_labels, classes=list(range(len(class_labels))))
                        else:
                            y_labels = label_binarize(true_labels, classes=class_labels)
                        if hasattr(clf, 'predict_proba'):
                            y_score = clf.predict_proba(features)
                            prob = True
                        elif hasattr(clf, 'decision_function'):
                            y_score = clf.decision_function(features)
                            prob = False
                        else:
                            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")

                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_score[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])

                        ## Compute micro-average ROC curve and ROC area
                        fpr["micro"], tpr["micro"], _ = roc_curve(y_labels.ravel(), y_score.ravel())
                        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                        ## Compute macro-average ROC curve and ROC area
                        # First aggregate all false positive rates
                        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                        # Then interpolate all ROC curves at this points
                        mean_tpr = np.zeros_like(all_fpr)
                        for i in range(n_classes):
                            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                        # Finally average it and compute AUC
                        mean_tpr /= n_classes
                        fpr["macro"] = all_fpr
                        tpr["macro"] = mean_tpr
                        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                        ## Plot ROC curves
                        plt.figure(figsize=(6, 4))
                        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:2.2%})'
                                                                   ''.format(roc_auc["micro"]), linewidth=3)

                        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:2.2%})'
                                                                   ''.format(roc_auc["macro"]), linewidth=3)

                        for i, label in enumerate(class_names):
                            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:2.2%})'
                                                           ''.format(label, roc_auc[i]), linewidth=2, linestyle=':')
                        roc_auc = roc_auc["macro"]
                    else:
                        raise ValueError('Number of classes should be atleast 2 or more')

                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([-0.01, 1.0])
                    plt.ylim([0.0, 1.01])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    plt.show()

                    return prob, y_score, roc_auc

            else:
                print("Invalid Option")

            print("\n")
            submenu2()
            option = int(input("\nInsert the command that you want to execute:"))
    elif option == 3:

        submenu3()
        option = int(input("\nInsert the command that you want to execute:\n"))

        while option != 0:
            if option == 1:
                submenu4()
                while option != 0:
                    if option == 1:
                        class_ql = {'low': 0, 'medium': 1, 'high': 2}
                        y_ql = wines_binary.quality.map(class_ql)

                        wqp_class_labels = np.array(wines_binary['quality'])
                        target_names = ['low', 'medium', 'high']

                        cols = wines_binary.columns
                        cols = list(cols.drop(['quality']))
                        X_train, X_test, y_train, y_test = train_test_split(wines.loc[:, cols], y_ql.values,
                                                                            test_size=0.20, random_state=101)

                        clf = Pipeline([
                            # ('pca', PCA(random_state = 101)),
                            ('clf', GaussianNB())])
                        param_grid = {}

                        gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1,
                                          n_jobs=-1)
                        NB = Pipeline([
                            # ('sel', select_fetaures(select_cols=SEL)),
                            ('scl', StandardScaler()),
                            ('gs', gs)
                        ])

                        NB.fit(X_train, y_train)

                        results = get_results(NB, 'NB', X_test, y_test,
                                              target_names=target_names, results=None, reasume=True)

                    elif option == 2:
                        cols = wines_binary.columns
                        cols = list(cols.drop(['quality']))

                        cols_clean = cols.copy()
                        cols_clean.remove('total_sulfur_dioxide')
                        cols_clean.remove('residual_sugar')


                        class select_fetaures(object):  # BaseEstimator, TransformerMixin,
                            def __init__(self, select_cols):
                                self.select_cols_ = select_cols

                            def fit(self, X, Y):
                                pass

                            def transform(self, X):
                                return X.loc[:, self.select_cols_]

                            def fit_transform(self, X, Y):
                                self.fit(X, Y)
                                df = self.transform(X)
                                return df

                            def __getitem__(self, x):
                                return self.X[x], self.Y[x]


                        clf = Pipeline([
                            # ('pca', PCA(random_state = 101)),
                            ('clf', KNeighborsClassifier())])

                        # a list of dictionaries to specify the parameters that we'd want to tune
                        SEL = cols_clean
                        n_components = [len(SEL) - 2, len(SEL) - 1, len(SEL)]
                        whiten = [True, False]

                        param_grid = \
                            [{'clf__n_neighbors': [10, 11, 12, 13]
                                 , 'clf__weights': ['distance']
                                 , 'clf__algorithm': ['ball_tree']  # , 'brute', 'auto',  'kd_tree', 'brute']
                                 , 'clf__leaf_size': [12, 11, 13]
                                 , 'clf__p': [1]
                              # ,'pca__n_components' : n_components
                              # ,'pca__whiten' : whiten
                              }]

                        gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1,
                                          n_jobs=-1)

                        KNNC = Pipeline([
                            ('sel', select_fetaures(select_cols=SEL)),
                            ('scl', StandardScaler()),
                            ('gs', gs)
                        ])

                        KNNC.fit(X_train, y_train)

                        results = get_results(KNNC, 'KNeighborsClassifier', X_test, y_test,
                                              target_names=target_names, results=results, reasume=False)

                    elif option == 3:

                        # Train_test split
                        X = wines_binary.iloc[:, 0:12].values
                        y = wines_binary.iloc[:, 12:13].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                        # Create a stratified 10-fold cross validation set
                        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

                        acc = np.zeros((10))
                        pre = np.zeros((10))
                        rec = np.zeros((10))
                        f1 = np.zeros((10))
                        i = 0
                        for train_index, val_index in strtfdKFold.split(X_train, y_train):
                            X_t, X_val = X_train[train_index], X_train[val_index]
                            y_t, y_val = y_train[train_index], y_train[val_index]
                            classifier_logistic = LogisticRegression(max_iter=5000)
                            classifier_logistic.fit(X_t, y_t)
                            yhat = classifier_logistic.predict(X_val)
                            acc[i] = metrics.accuracy_score(yhat, y_val)
                            pre[i] = metrics.precision_score(yhat, y_val)
                            rec[i] = metrics.recall_score(yhat, y_val)
                            f1[i] = metrics.f1_score(yhat, y_val)
                            i = i + 1

                        print('\n')
                        print('Predictive measures for training data')
                        print('Mean accuracy: ' + str(np.mean(acc, axis=0)))
                        print("Mean precision: " + str(np.mean(pre, axis=0)))
                        print("Mean recall: " + str(np.mean(rec, axis=0)))
                        print("Mean f1-score: " + str(np.mean(f1, axis=0)))
                        print('\n')

                        classifier_logistic = LogisticRegression(max_iter=5000)
                        classifier_logistic.fit(X_train, y_train)
                        # Test the model with the test set
                        pred = classifier_logistic.predict(X_test)
                        print('Predictive measures for test data')
                        print('Test accuracy: ' + str(metrics.accuracy_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.precision_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.recall_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.f1_score(pred, y_test)))

                    elif option == 4:
                        class_ql = {'low': 0, 'medium': 1, 'high': 2}
                        y_ql = wines_binary.quality.map(class_ql)

                        wqp_class_labels = np.array(wines_binary['quality'])
                        target_names = ['low', 'medium', 'high']

                        cols = wines.columns
                        cols = list(cols.drop(['quality']))
                        X_train, X_test, y_train, y_test = train_test_split(wines_binary.loc[:, cols], y_ql.values,
                                                                            test_size=0.20, random_state=101)

                        clf = Pipeline([
                            ('clf', DecisionTreeClassifier(random_state=101))])

                        # a list of dictionaries to specify the parameters that we'd want to tune
                        criterion = ['gini', 'entropy']
                        splitter = ['best']
                        max_depth = [8, 9, 10, 11]  # [15, 20, 25]
                        min_samples_leaf = [2, 3, 5]
                        class_weight = ['balanced', None]

                        param_grid = \
                            [{'clf__class_weight': class_weight
                                 , 'clf__criterion': criterion
                                 , 'clf__splitter': splitter
                                 , 'clf__max_depth': max_depth
                                 , 'clf__min_samples_leaf': min_samples_leaf
                              }]

                        gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1,
                                          n_jobs=-1)
                        DT = Pipeline([
                            ('scl', StandardScaler()),
                            ('gs', gs)
                        ])

                        DT.fit(X_train, y_train)

                        results = get_results(DT, 'DT First', X_test, y_test, target_names=target_names, reasume=True)

                    elif option == 5:

                        from sklearn import tree
                        from sklearn.metrics import confusion_matrix
                        from sklearn.model_selection import train_test_split

                        print("White Wine")
                        X = whitewines.iloc[:, 0:11].values
                        y = whitewines.iloc[:, 11:12].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                        clf = tree.DecisionTreeClassifier()
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        print(clf.score(X_test, y_test) * 100, "%")
                        print(confusion_matrix(y_test, y_pred))

                        print("Red Wine")
                        X = redwines.iloc[:, 0:11].values
                        y = redwines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                        clf = tree.DecisionTreeClassifier()
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        print(clf.score(X_test, y_test) * 100, "%")
                        print(confusion_matrix(y_test, y_pred))

                    elif option == 6:
                        X = wines_binary_norm.iloc[:, 0:12].values
                        y = wines_binary_norm.iloc[:, 12:13].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                        print("\nShape of X_train: ", X_train.shape)
                        print("Shape of X_test: ", X_test.shape)
                        print("Shape of y_train: ", y_train.shape)
                        print("Shape of y_test", y_test.shape)

                        # Feature Scaling
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)

                        ann = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000)
                        ann.fit(X_train, y_train)
                        y_pred = ann.predict(X_test)
                        print("\nArtifial Neural Network")
                        print(confusion_matrix(y_test, y_pred))

                        print("\nNumber of well predicted points out of a total %d points: %d" % (
                            X_test.shape[0], (y_test == y_pred).sum()))
                        accuracy = ((y_test == y_pred).sum() / X_test.shape[0]) * 100
                        print("Precision = {:.2f} %".format(accuracy))

                        ann1 = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000)
                        ann1.fit(X_train_scaled, y_train.ravel())
                        # Predicting Cross Validation Score
                        cv_ann = cross_val_score(estimator=ann1, X=X_train_scaled, y=y_train.ravel(), cv=10)
                        print("CV: ", cv_rf.mean())

                        y_pred_ann_train = ann1.predict(X_train_scaled)
                        accuracy_ann_train = accuracy_score(y_train, y_pred_ann_train)
                        print("Training set: ", accuracy_ann_train)

                        y_pred_ann_test = ann1.predict(X_test_scaled)
                        accuracy_ann_test = accuracy_score(y_test, y_pred_ann_test)
                        print("Test set: ", accuracy_ann_test)

                        tp_ann = confusion_matrix(y_test, y_pred_ann_test)[0, 0]
                        fp_ann = confusion_matrix(y_test, y_pred_ann_test)[0, 1]
                        tn_ann = confusion_matrix(y_test, y_pred_ann_test)[1, 1]
                        fn_ann = confusion_matrix(y_test, y_pred_ann_test)[1, 0]

                    elif option == 7:

                        # Train_test split
                        X = wines_binary_norm.iloc[:, 0:12].values
                        y = wines_binary_norm.iloc[:, 12:13].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                        # Create a stratified 10-fold cross validation set
                        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

                        acc = np.zeros((10))
                        pre = np.zeros((10))
                        rec = np.zeros((10))
                        f1 = np.zeros((10))
                        i = 0
                        for train_index, val_index in strtfdKFold.split(X_train, y_train):
                            X_t, X_val = X_train[train_index], X_train[val_index]
                            y_t, y_val = y_train[train_index], y_train[val_index]
                            classifier_svm = svm.SVC(kernel='linear')
                            classifier_svm.fit(X_t, y_t)
                            yhat = classifier_svm.predict(X_val)
                            acc[i] = metrics.accuracy_score(yhat, y_val)
                            pre[i] = metrics.precision_score(yhat, y_val)
                            rec[i] = metrics.recall_score(yhat, y_val)
                            f1[i] = metrics.f1_score(yhat, y_val)
                            i = i + 1

                        print('\n Kernel is linear')
                        print('Mean accuracy: ' + str(np.mean(acc, axis=0)))
                        print("Mean precision: " + str(np.mean(pre, axis=0)))
                        print("Mean recall: " + str(np.mean(rec, axis=0)))
                        print("Mean f1-score: " + str(np.mean(f1, axis=0)))

                        classifier_svm = classifier_svm = svm.SVC(kernel='linear')
                        classifier_svm.fit(X_train, y_train)
                        # Test the model with the test set
                        pred = classifier_svm.predict(X_test)
                        print('Test accuracy: ' + str(metrics.accuracy_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.precision_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.recall_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.f1_score(pred, y_test)))

                        # Train_test split
                        X = wines_binary_norm.iloc[:, 0:12].values
                        y = wines_binary_norm.iloc[:, 12:13].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                        # Create a stratified 10-fold cross validation set
                        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

                        acc = np.zeros((10))
                        pre = np.zeros((10))
                        rec = np.zeros((10))
                        f1 = np.zeros((10))
                        i = 0
                        for train_index, val_index in strtfdKFold.split(X_train, y_train):
                            X_t, X_val = X_train[train_index], X_train[val_index]
                            y_t, y_val = y_train[train_index], y_train[val_index]
                            classifier_svm = svm.SVC(kernel='poly')
                            classifier_svm.fit(X_t, y_t)
                            yhat = classifier_svm.predict(X_val)
                            acc[i] = metrics.accuracy_score(yhat, y_val)
                            pre[i] = metrics.precision_score(yhat, y_val)
                            rec[i] = metrics.recall_score(yhat, y_val)
                            f1[i] = metrics.f1_score(yhat, y_val)
                            i = i + 1

                        print('\n Kernel is polynomial')
                        print('Mean accuracy: ' + str(np.mean(acc, axis=0)))
                        print("Mean precision: " + str(np.mean(pre, axis=0)))
                        print("Mean recall: " + str(np.mean(rec, axis=0)))
                        print("Mean f1-score: " + str(np.mean(f1, axis=0)))

                        classifier_svm = classifier_svm = svm.SVC(kernel='poly')
                        classifier_svm.fit(X_train, y_train)
                        # Test the model with the test set
                        pred = classifier_svm.predict(X_test)
                        print('Test accuracy: ' + str(metrics.accuracy_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.precision_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.recall_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.f1_score(pred, y_test)))

                        # Train_test split
                        X = wines_binary_norm.iloc[:, 0:12].values
                        y = wines_binary_norm.iloc[:, 12:13].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                        # Create a stratified 10-fold cross validation set
                        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

                        acc = np.zeros((10))
                        pre = np.zeros((10))
                        rec = np.zeros((10))
                        f1 = np.zeros((10))
                        i = 0
                        for train_index, val_index in strtfdKFold.split(X_train, y_train):
                            X_t, X_val = X_train[train_index], X_train[val_index]
                            y_t, y_val = y_train[train_index], y_train[val_index]
                            classifier_svm = svm.SVC(kernel='sigmoid')
                            classifier_svm.fit(X_t, y_t)
                            yhat = classifier_svm.predict(X_val)
                            acc[i] = metrics.accuracy_score(yhat, y_val)
                            pre[i] = metrics.precision_score(yhat, y_val)
                            rec[i] = metrics.recall_score(yhat, y_val)
                            f1[i] = metrics.f1_score(yhat, y_val)
                            i = i + 1

                        print('\n Kernel is sigmoid')
                        print('Mean accuracy: ' + str(np.mean(acc, axis=0)))
                        print("Mean precision: " + str(np.mean(pre, axis=0)))
                        print("Mean recall: " + str(np.mean(rec, axis=0)))
                        print("Mean f1-score: " + str(np.mean(f1, axis=0)))

                        classifier_svm = classifier_svm = svm.SVC(kernel='sigmoid')
                        classifier_svm.fit(X_train, y_train)
                        # Test the model with the test set
                        pred = classifier_svm.predict(X_test)
                        print('Test accuracy: ' + str(metrics.accuracy_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.precision_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.recall_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.f1_score(pred, y_test)))

                        # Train_test split
                        X = wines_binary_norm.iloc[:, 0:12].values
                        y = wines_binary_norm.iloc[:, 12:13].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                        # Create a stratified 10-fold cross validation set
                        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

                        acc = np.zeros((10))
                        pre = np.zeros((10))
                        rec = np.zeros((10))
                        f1 = np.zeros((10))
                        i = 0
                        for train_index, val_index in strtfdKFold.split(X_train, y_train):
                            X_t, X_val = X_train[train_index], X_train[val_index]
                            y_t, y_val = y_train[train_index], y_train[val_index]
                            classifier_svm = svm.SVC(kernel='rbf')
                            classifier_svm.fit(X_t, y_t)
                            yhat = classifier_svm.predict(X_val)
                            acc[i] = metrics.accuracy_score(yhat, y_val)
                            pre[i] = metrics.precision_score(yhat, y_val)
                            rec[i] = metrics.recall_score(yhat, y_val)
                            f1[i] = metrics.f1_score(yhat, y_val)
                            i = i + 1

                        print('\n Kernel is rbf')
                        print('Mean accuracy: ' + str(np.mean(acc, axis=0)))
                        print("Mean precision: " + str(np.mean(pre, axis=0)))
                        print("Mean recall: " + str(np.mean(rec, axis=0)))
                        print("Mean f1-score: " + str(np.mean(f1, axis=0)))

                        classifier_svm = classifier_svm = svm.SVC(kernel='rbf')
                        classifier_svm.fit(X_train, y_train)
                        # Test the model with the test set
                        pred = classifier_svm.predict(X_test)
                        print('Test accuracy: ' + str(metrics.accuracy_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.precision_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.recall_score(pred, y_test)))
                        print('Test accuracy: ' + str(metrics.f1_score(pred, y_test)))

                    elif option == 8:

                        class_ql = {'low': 0, 'medium': 1, 'high': 2}
                        y_ql = wines_binary.quality.map(class_ql)

                        wqp_class_labels = np.array(wines_binary['quality'])
                        target_names = ['low', 'medium', 'high']

                        cols = wines_binary.columns
                        cols = list(cols.drop(['quality']))
                        X_train, X_test, y_train, y_test = train_test_split(wines.loc[:, cols], y_ql.values,
                                                                            test_size=0.20, random_state=101)

                        clf = Pipeline([
                            # ('pca', PCA(random_state = 101)),
                            ('clf', RandomForestClassifier(random_state=101))])

                        # a list of dictionaries to specify the parameters that we'd want to tune
                        SEL = cols
                        n_components = [len(SEL) - 2, len(SEL) - 1, len(SEL)]
                        whiten = [True, False]
                        criterion = ['gini', 'entropy']
                        class_weight = ['balanced', None]
                        n_estimators = [155, 175]
                        max_depth = [20, None]  # , 3, 4, 5, 10] #
                        min_samples_split = [2, 3, 4]
                        min_samples_leaf = [1]  # , 2 , 3]

                        param_grid = \
                            [{  # 'clf__class_weight': class_weight
                                'clf__criterion': criterion
                                , 'clf__n_estimators': n_estimators
                                , 'clf__min_samples_split': min_samples_split
                                , 'clf__max_depth': max_depth
                                # ,'clf__min_samples_leaf': min_samples_leaf
                                # ,'pca__n_components' : n_components
                                # ,'pca__whiten' : whiten
                            }]

                        gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1,
                                          n_jobs=-1)
                        RF = Pipeline([
                            # ('sel', select_fetaures(select_cols=SEL)),
                            ('scl', StandardScaler()),
                            ('gs', gs)
                        ])

                        RF.fit(X_train, y_train)

                        results = get_results(RF, 'RF', X_test, y_test,
                                              target_names=target_names, results=None, reasume=True)

                    elif option == 10:
                        from sklearn import tree

                        print("\n ")
                        print("\n Red Wine")
                        X = redwines.iloc[:, 0:11].values
                        y = redwines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                        print("NaiveBayes")
                        gnb = GaussianNB()
                        y_pred = gnb.fit(X_train, y_train).predict(X_test)
                        print(confusion_matrix(y_test, y_pred))
                        print("Number of mislabeled points out of a total %d points : %d" % (
                            X_test.shape[0], (y_test != y_pred).sum()))

                        accuracynb = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                        print(accuracynb, "%")

                        print("KNN")
                        neigh = KNeighborsClassifier(n_neighbors=48)
                        y_pred = neigh.fit(X_train, y_train).predict(X_test)
                        print(confusion_matrix(y_test, y_pred))
                        print("Number of mislabeled points out of a total %d points : %d" % (
                            X_test.shape[0], (y_test != y_pred).sum()))

                        accuracyknn = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                        print(accuracyknn, "%")

                        print("Decision Tree")
                        clf = tree.DecisionTreeClassifier()
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        accuracydt = clf.score(X_test, y_test) * 100
                        print(accuracydt, "%")
                        print(confusion_matrix(y_test, y_pred))

                        print("RandomForest")
                        rf = RandomForestClassifier(max_depth=10, random_state=0)
                        clf = rf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        accuracyrf = clf.score(X_test, y_test) * 100
                        print(accuracyrf, "%")
                        print(confusion_matrix(y_test, y_pred))

                        print("K-Means Clustering")
                        rf = k_means = cluster.KMeans(n_clusters=2)
                        clf = rf.fit(redwines)
                        centroids = clf.cluster_centers_
                        accuracykmc = silhouette_score(redwines, clf.labels_) * 100
                        print(centroids)
                        print(accuracykmc, "%")

                        print("Perceptron Classification")

                        scaler = MinMaxScaler()
                        scaler.fit(redwines.iloc[:, 0:11])
                        scaled_features = scaler.transform(redwines.iloc[:, 0:11])
                        print(scaled_features)

                        # volta a converter em dataframe
                        print('\n')
                        new = pd.DataFrame(data=scaled_features,
                                           columns=["fixed_acidity",
                                                    "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                                    "free_sulfur_dioxide",
                                                    "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"])
                        print(new)

                        X = new.iloc[:, 0:11].values
                        y = redwines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        clf = Perceptron(verbose=3)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        accuracypc = clf.score(X_test, y_test) * 100
                        print(accuracypc)

                        models = [('K-Nearest Neighbors (KNN)', accuracyknn),
                                  ('Naive Bayes', accuracynb),
                                  ('Decision Tree Classification', accuracydt),
                                  ('Random Forest Tree Classification', accuracyrf),
                                  ('K-Means Clustering', accuracykmc),
                                  ("Perceptron Classification", accuracypc)]

                        predict = pd.DataFrame(data=models, columns=['Model', 'Accuracy'])
                        print(predict)

                        f, axe = plt.subplots(1, 1, figsize=(18, 6))

                        predict.sort_values(by=['Accuracy'], ascending=False, inplace=True)

                        sns.barplot(x='Accuracy', y='Model', data=predict, ax=axe)
                        # axes[0].set(xlabel='Region', ylabel='Charges')
                        axe.set_xlabel('Score in %)', size=16)
                        axe.set_ylabel('Model')

                        plt.show()
                    elif option == 11:

                        X = whitewines.iloc[:, 0:11].values
                        y = whitewines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        print("KNN")
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)
                        classifier_knn = KNeighborsClassifier(leaf_size=1, metric='minkowski', n_neighbors=32,
                                                              weights='distance')
                        classifier_knn.fit(X_train_scaled, y_train.ravel())
                        cv_knn = cross_val_score(estimator=classifier_knn, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_knn)
                        print("MEAN:", cv_knn.mean())

                        print("\n")
                        print("NaiveBayes")
                        classifier_nb = GaussianNB()
                        classifier_nb.fit(X_train_scaled, y_train.ravel())

                        # Predicting Cross Validation Score
                        cv_nb = cross_val_score(estimator=classifier_nb, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_nb)
                        print("MEAN:", cv_nb.mean())

                        print("\n")
                        print("Decision Tree")
                        from sklearn.tree import DecisionTreeClassifier

                        classifier_dt = DecisionTreeClassifier(criterion='gini', max_features=6, max_leaf_nodes=400,
                                                               random_state=33)
                        classifier_dt.fit(X_train_scaled, y_train.ravel())
                        # Predicting Cross Validation Score
                        cv_dt = cross_val_score(estimator=classifier_dt, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_dt)
                        print("MEAN:", cv_dt.mean())

                        print("\n")
                        print("Random Forest Classification")
                        from sklearn.ensemble import RandomForestClassifier

                        classifier_rf = RandomForestClassifier(criterion='entropy', max_features=4, n_estimators=800,
                                                               random_state=33)
                        classifier_rf.fit(X_train_scaled, y_train.ravel())

                        # Predicting Cross Validation Score
                        cv_rf = cross_val_score(estimator=classifier_rf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_rf)
                        print("MEAN:", cv_rf.mean())

                        print("\n")
                        print("K-Means Clustering")

                        print("\n")
                        print("Perceptron Classification")
                        clf = Perceptron(verbose=3)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        # Predicting Cross Validation Score
                        cv_pc = cross_val_score(estimator=clf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_pc)
                        print("MEAN:", cv_pc.mean())

                        tableScores = {
                            'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
                                      'Random Forest Tree Classification', 'Perceptron Classification'],
                            'CV1': [cv_knn[0], cv_nb[0], cv_dt[0], cv_rf[0], cv_pc[0]],
                            'CV2': [cv_knn[1], cv_nb[1], cv_dt[1], cv_rf[1], cv_pc[1]],
                            'CV3': [cv_knn[2], cv_nb[2], cv_dt[2], cv_rf[2], cv_pc[2]],
                            'CV4': [cv_knn[3], cv_nb[3], cv_dt[3], cv_rf[3], cv_pc[3]],
                            'CV5': [cv_knn[4], cv_nb[4], cv_dt[4], cv_rf[4], cv_pc[4]],
                            'Media': [cv_knn.mean(), cv_nb.mean(), cv_dt.mean(), cv_rf.mean(), cv_pc.mean()]}

                        dfScores = pd.DataFrame(tableScores)
                        # Create a column Rating_Rank which contains
                        # the rank of each movie based on rating
                        dfScores['Ranking'] = dfScores['Media'].rank(ascending=0)

                        # Set the index to newly created column, Rating_Rank
                        dfScores = dfScores.set_index('Ranking')
                        dff = dfScores.sort_index()
                        print(dff)

                    elif option == 12:
                        X = redwines.iloc[:, 0:11].values
                        y = redwines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        print("KNN")
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)
                        classifier_knn = KNeighborsClassifier(leaf_size=1, metric='minkowski', n_neighbors=32,
                                                              weights='distance')
                        classifier_knn.fit(X_train_scaled, y_train.ravel())
                        cv_knn = cross_val_score(estimator=classifier_knn, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_knn)
                        print("MEAN:", cv_knn.mean())

                        print("\n")
                        print("NaiveBayes")
                        classifier_nb = GaussianNB()
                        classifier_nb.fit(X_train_scaled, y_train.ravel())

                        # Predicting Cross Validation Score
                        cv_nb = cross_val_score(estimator=classifier_nb, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_nb)
                        print("MEAN:", cv_nb.mean())

                        print("\n")
                        print("Decision Tree")
                        from sklearn.tree import DecisionTreeClassifier

                        classifier_dt = DecisionTreeClassifier(criterion='gini', max_features=6, max_leaf_nodes=400,
                                                               random_state=33)
                        classifier_dt.fit(X_train_scaled, y_train.ravel())
                        # Predicting Cross Validation Score
                        cv_dt = cross_val_score(estimator=classifier_dt, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_dt)
                        print("MEAN:", cv_dt.mean())

                        print("\n")
                        print("Random Forest Classification")
                        from sklearn.ensemble import RandomForestClassifier

                        classifier_rf = RandomForestClassifier(criterion='entropy', max_features=4, n_estimators=800,
                                                               random_state=33)
                        classifier_rf.fit(X_train_scaled, y_train.ravel())

                        # Predicting Cross Validation Score
                        cv_rf = cross_val_score(estimator=classifier_rf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_rf)
                        print("MEAN:", cv_rf.mean())

                        print("\n")
                        print("K-Means CLustering")

                        print("\n")
                        print("Perceptron Classification")
                        clf = Perceptron(verbose=3)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        # Predicting Cross Validation Score
                        cv_pc = cross_val_score(estimator=clf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                        print("CV:", cv_pc)
                        print("MEAN:", cv_pc.mean())

                        # d = {'K-Nearest Neighbors (KNN)':cv_knn  , 'Naive Bayes': cv_knn, 'Decision Tree Classification': cv_dt, 'Random Forest Tree Classification': cv_rf , 'Perceptron Classification' : cv_pc}
                        # predict = pd.DataFrame(data=d)
                        # print(predict)

                        tableScores = {
                            'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
                                      'Random Forest Tree Classification', 'Perceptron Classification'],
                            'CV1': [cv_knn[0], cv_nb[0], cv_dt[0], cv_rf[0], cv_pc[0]],
                            'CV2': [cv_knn[1], cv_nb[1], cv_dt[1], cv_rf[1], cv_pc[1]],
                            'CV3': [cv_knn[2], cv_nb[2], cv_dt[2], cv_rf[2], cv_pc[2]],
                            'CV4': [cv_knn[3], cv_nb[3], cv_dt[3], cv_rf[3], cv_pc[3]],
                            'CV5': [cv_knn[4], cv_nb[4], cv_dt[4], cv_rf[4], cv_pc[4]],
                            'Mean': [cv_knn.mean(), cv_nb.mean(), cv_dt.mean(), cv_rf.mean(), cv_pc.mean()]}

                        dfScores = pd.DataFrame(tableScores)
                        # Create a column Rating_Rank which contains
                        # the rank of each movie based on rating
                        dfScores['Ranking'] = dfScores['Mean'].rank(ascending=0)

                        # Set the index to newly created column, Rating_Rank
                        dfScores = dfScores.set_index('Ranking')
                        dff = dfScores.sort_index()
                        print(dff)

                    elif option == 13:
                        from sklearn import cluster
                        from sklearn.metrics import silhouette_score

                        X = whitewines.iloc[:, 0:11].values
                        y = whitewines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        rf = k_means = cluster.KMeans(n_clusters=2)
                        clf = rf.fit(whitewines)
                        centroids = clf.cluster_centers_
                        score = silhouette_score(whitewines, clf.labels_)
                        print(centroids)
                        print(score)

                    elif option == 14:
                        from sklearn import cluster
                        from sklearn.metrics import silhouette_score

                        X = redwines.iloc[:, 0:11].values
                        y = redwines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        rf = k_means = cluster.KMeans(n_clusters=2)
                        clf = rf.fit(redwines)
                        centroids = clf.cluster_centers_
                        score = silhouette_score(redwines, clf.labels_)
                        print(centroids)
                        print(score)

                    elif option == 15:
                        import pandas as pd
                        from matplotlib import pyplot as plt
                        from scipy.cluster.hierarchy import dendrogram
                        from seaborn import scatterplot
                        from sklearn.cluster import AgglomerativeClustering
                        from sklearn.datasets import make_blobs
                        import numpy as np

                        nb_samples = 3000
                        X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=8, cluster_std=2.0)
                        ac = AgglomerativeClustering(n_clusters=8, linkage='complete')
                        Y = ac.fit_predict(X)
                        df = pd.DataFrame({" a ": X[:, 0], " b ": X[:, 1], " c ": Y})
                        plt.title('Hierarchical Clustering Dendrogram')
                        scatterplot(data=df, x=" a ", y=" b ", hue=" c ", palette=" deep ")
                        plt.show()


                        def plot_dendrogram(model, **kwargs):
                            counts = np.zeros(model.children_.shape[0])
                            n_samples = len(model.labels_)
                            for i, merge in enumerate(model.children_):
                                current_count = 0
                                for child_idx in merge:
                                    if child_idx < n_samples:
                                        current_count += 1
                                    else:
                                        current_count += counts[child_idx - n_samples]
                                        counts[i] = current_count
                            linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
                            dendrogram(linkage_matrix, **kwargs)

                            nb_samples = 3000
                            X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=8, cluster_std=2.0)
                            ac = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
                            Y = ac.fit_predict(X)
                            plt.title('Hierarchical Clustering  Dendrogram')
                            plot_dendrogram(ac, truncate_mode='level', p=3)
                            plt.xlabel("Number of points in node (or index of point if  no parenthesis).")
                            plt.show()

                    elif option == 16:

                        from sklearn.linear_model import Perceptron
                        from sklearn.metrics import confusion_matrix
                        from sklearn.model_selection import train_test_split

                        scaler = MinMaxScaler()
                        scaler.fit(redwines.iloc[:, 0:11])
                        scaled_features = scaler.transform(redwines.iloc[:, 0:11])
                        print(scaled_features)

                        # volta a converter em dataframe
                        print('\n')
                        new = pd.DataFrame(data=scaled_features,
                                           columns=["fixed_acidity",
                                                    "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                                    "free_sulfur_dioxide",
                                                    "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"])
                        print(new)

                        X = new.iloc[:, 0:11].values
                        y = redwines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        clf = Perceptron(verbose=3)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        print(clf.score(X_test, y_test))
                        print(confusion_matrix(y_test, y_pred))

                    elif option == 17:
                        from sklearn.linear_model import Perceptron
                        from sklearn.metrics import confusion_matrix
                        from sklearn.model_selection import train_test_split

                        scaler = MinMaxScaler()
                        scaler.fit(whitewines.iloc[:, 0:11])
                        scaled_features = scaler.transform(whitewines.iloc[:, 0:11])
                        print(scaled_features)

                        # volta a converter em dataframe
                        print('\n')
                        new = pd.DataFrame(data=scaled_features,
                                           columns=["fixed_acidity",
                                                    "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                                    "free_sulfur_dioxide",
                                                    "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"])
                        print(new)

                        X = new.iloc[:, 0:11].values
                        y = whitewines.iloc[:, 11:12].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        clf = Perceptron(verbose=3)
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        print(clf.score(X_test, y_test))
                        print(confusion_matrix(y_test, y_pred))

                    elif option == 18:
                        def naiveBayes():

                            print("\n")
                            print("\n Red Wine")

                            X = redwines.iloc[:, 0:11].values
                            y = redwines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                            print("NaiveBayes")
                            gnb = GaussianNB()
                            y_pred = gnb.fit(X_train, y_train).predict(X_test)

                            cf_matrix = confusion_matrix(y_test, y_pred)
                            print(cf_matrix)

                            plt.show()

                            print("Number of mislabeled points out of a total %d points : %d" % (
                                X_test.shape[0], (y_test != y_pred).sum()))

                            accuracy = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                            print(accuracy, "%")


                        def knn():
                            print("\n")
                            print("\nRed Wine")

                            X = redwines.iloc[:, 0:11].values
                            y = redwines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                            neigh = KNeighborsClassifier(n_neighbors=25)
                            y_pred = neigh.fit(X_train, y_train).predict(X_test)

                            print(confusion_matrix(y_test, y_pred))
                            print("Number of mislabeled points out of a total %d points : %d" % (
                                X_test.shape[0], (y_test != y_pred).sum()))

                            accuracy = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                            print(accuracy, "%")


                        def dt():
                            from sklearn import tree

                            print("Red Wine")
                            X = redwines.iloc[:, 0:11].values
                            y = redwines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                            clf = tree.DecisionTreeClassifier()
                            clf = clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            print(clf.score(X_test, y_test) * 100, "%")
                            print(confusion_matrix(y_test, y_pred))


                        def rf():
                            print("\n")
                            print("Red Wine")
                            X = redwines.iloc[:, 0:11].values
                            y = redwines.iloc[:, 11:12].values.ravel()
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                            rf = RandomForestClassifier(max_depth=10, random_state=0)
                            clf = rf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            print(clf.score(X_test, y_test) * 100, "%")
                            print(confusion_matrix(y_test, y_pred))


                        def k_means():

                            from sklearn import cluster
                            from sklearn.metrics import silhouette_score

                            X = redwines.iloc[:, 0:11].values
                            y = redwines.iloc[:, 11:12].values.ravel()
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                            rf = k_means = cluster.KMeans(n_clusters=2)
                            clf = rf.fit(redwines)
                            centroids = clf.cluster_centers_
                            score = silhouette_score(redwines, clf.labels_)
                            print(centroids)
                            print(score)


                        def perceptron():
                            from sklearn.linear_model import Perceptron
                            from sklearn.metrics import confusion_matrix
                            from sklearn.model_selection import train_test_split

                            scaler = MinMaxScaler()
                            scaler.fit(redwines.iloc[:, 0:11])
                            scaled_features = scaler.transform(redwines.iloc[:, 0:11])
                            print(scaled_features)

                            # volta a converter em dataframe
                            print('\n')
                            new = pd.DataFrame(data=scaled_features,
                                               columns=["fixed_acidity",
                                                        "volatile_acidity", "citric_acid", "residual_sugar",
                                                        "chlorides",
                                                        "free_sulfur_dioxide",
                                                        "total_sulfur_dioxide", "density", "pH", "sulphates",
                                                        "alcohol"])
                            print(new)

                            X = new.iloc[:, 0:11].values
                            y = redwines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                            clf = Perceptron(verbose=3)
                            clf = clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            print(clf.score(X_test, y_test))
                            print(confusion_matrix(y_test, y_pred))


                        import datetime

                        starting_timenb = datetime.datetime.now()
                        naiveBayes()
                        end_timenb = datetime.datetime.now()
                        difnaiveBayes = end_timenb - starting_timenb

                        starting_timeknn = datetime.datetime.now()
                        knn()
                        end_timeknn = datetime.datetime.now()
                        difknn = end_timeknn - starting_timeknn

                        starting_timedt = datetime.datetime.now()
                        dt()
                        end_timedt = datetime.datetime.now()
                        difdt = end_timedt - starting_timedt

                        starting_timerf = datetime.datetime.now()
                        rf()
                        end_timerf = datetime.datetime.now()
                        difrf = end_timerf - starting_timerf

                        starting_timek_means = datetime.datetime.now()
                        k_means()
                        end_timek_means = datetime.datetime.now()
                        difk_means = end_timek_means - starting_timek_means

                        starting_timeperceptron = datetime.datetime.now()
                        perceptron()
                        end_timeperceptron = datetime.datetime.now()
                        difperceptron = end_timeperceptron - starting_timeperceptron

                        print(difnaiveBayes, difknn, difdt, difrf, difk_means, difperceptron)
                        print("\n")

                        tableScores = {
                            'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
                                      'Random Forest Tree Classification', 'Perceptron Classification',
                                      'K-Means Clustering'],
                            'Time': [difknn, difnaiveBayes, difdt, difrf, difperceptron, difk_means], }

                        dfScores = pd.DataFrame(tableScores)
                        # Create a column Rating_Rank which contains
                        # the rank of each movie based on rating
                        dfScores['Ranking'] = dfScores['Time'].rank(ascending=1)

                        # Set the index to newly created column, Rating_Rank
                        dfScores = dfScores.set_index('Ranking')
                        dff = dfScores.sort_index()
                        print(dff)

                        tabelatempos = {
                            'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
                                      'Random Forest Tree Classification', 'Perceptron Classification',
                                      'K-Means Clustering'],
                            'Time': [0.038773, 0.007979, 0.009033, 0.258831, 0.031914, 0.142420], }

                        dftempo = pd.DataFrame(tabelatempos)
                        sns.barplot(x=dftempo['Model'], y=dftempo['Time'], data=pd.melt(dftempo))
                        plt.title("Execution time of the different models")
                        plt.show()

                    elif option == 23:
                        def naiveBayes():

                            print("\n ")
                            print("\n White Wine")

                            X = whitewines.iloc[:, 0:11].values
                            y = whitewines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                            print("NaiveBayes")
                            gnb = GaussianNB()
                            y_pred = gnb.fit(X_train, y_train).predict(X_test)

                            cf_matrix = confusion_matrix(y_test, y_pred)
                            print(cf_matrix)

                            plt.show()

                            print("Number of mislabeled points out of a total %d points : %d" % (
                                X_test.shape[0], (y_test != y_pred).sum()))

                            accuracy = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                            print(accuracy, "%")


                        def knn():
                            print("\n ")
                            print("\n White Wine")

                            X = whitewines.iloc[:, 0:11].values
                            y = whitewines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                            neigh = KNeighborsClassifier(n_neighbors=25)
                            y_pred = neigh.fit(X_train, y_train).predict(X_test)

                            print(confusion_matrix(y_test, y_pred))
                            print("Number of mislabeled points out of a total %d points : %d" % (
                                X_test.shape[0], (y_test != y_pred).sum()))

                            accuracy = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                            print(accuracy, "%")


                        def dt():
                            from sklearn import tree

                            print("White Wine")
                            X = whitewines.iloc[:, 0:11].values
                            y = whitewines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                            clf = tree.DecisionTreeClassifier()
                            clf = clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            print(clf.score(X_test, y_test) * 100, "%")
                            print(confusion_matrix(y_test, y_pred))


                        def rf():
                            print("\n")
                            print("White Wine")
                            X = whitewines.iloc[:, 0:11].values
                            y = whitewines.iloc[:, 11:12].values.ravel()
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                            rf = RandomForestClassifier(max_depth=10, random_state=0)
                            clf = rf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            print(clf.score(X_test, y_test) * 100, "%")
                            print(confusion_matrix(y_test, y_pred))


                        def k_means():
                            from sklearn import cluster
                            from sklearn.metrics import silhouette_score

                            X = whitewines.iloc[:, 0:11].values
                            y = whitewines.iloc[:, 11:12].values.ravel()
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                            rf = k_means = cluster.KMeans(n_clusters=2)
                            clf = rf.fit(redwines)
                            centroids = clf.cluster_centers_
                            score = silhouette_score(redwines, clf.labels_)
                            print(centroids)
                            print(score)


                        def perceptron():
                            from sklearn.linear_model import Perceptron
                            from sklearn.metrics import confusion_matrix
                            from sklearn.model_selection import train_test_split

                            scaler = MinMaxScaler()
                            scaler.fit(whitewines.iloc[:, 0:11])
                            scaled_features = scaler.transform(whitewines.iloc[:, 0:11])
                            print(scaled_features)

                            # volta a converter em dataframe
                            print('\n')
                            new = pd.DataFrame(data=scaled_features,
                                               columns=["fixed_acidity",
                                                        "volatile_acidity", "citric_acid", "residual_sugar",
                                                        "chlorides",
                                                        "free_sulfur_dioxide",
                                                        "total_sulfur_dioxide", "density", "pH", "sulphates",
                                                        "alcohol"])
                            print(new)

                            X = new.iloc[:, 0:11].values
                            y = whitewines.iloc[:, 11:12].values.ravel()

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                            clf = Perceptron(verbose=3)
                            clf = clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            print(clf.score(X_test, y_test))
                            print(confusion_matrix(y_test, y_pred))


                        import datetime

                        starting_timenb = datetime.datetime.now()
                        naiveBayes()
                        end_timenb = datetime.datetime.now()
                        difnaiveBayes = end_timenb - starting_timenb

                        starting_timeknn = datetime.datetime.now()
                        knn()
                        end_timeknn = datetime.datetime.now()
                        difknn = end_timeknn - starting_timeknn

                        starting_timedt = datetime.datetime.now()
                        dt()
                        end_timedt = datetime.datetime.now()
                        difdt = end_timedt - starting_timedt

                        starting_timerf = datetime.datetime.now()
                        rf()
                        end_timerf = datetime.datetime.now()
                        difrf = end_timerf - starting_timerf

                        starting_timek_means = datetime.datetime.now()
                        k_means()
                        end_timek_means = datetime.datetime.now()
                        difk_means = end_timek_means - starting_timek_means

                        starting_timeperceptron = datetime.datetime.now()
                        perceptron()
                        end_timeperceptron = datetime.datetime.now()
                        difperceptron = end_timeperceptron - starting_timeperceptron

                        print(difnaiveBayes, difknn, difdt, difrf, difk_means, difperceptron)

                        tableScores = {
                            'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
                                      'Random Forest Tree Classification', 'Perceptron Classification',
                                      'K-Means Clustering'],
                            'Time': [difknn, difnaiveBayes, difdt, difrf, difperceptron, difk_means], }

                        dfScores = pd.DataFrame(tableScores)
                        # Create a column Rating_Rank which contains
                        # the rank of each movie based on rating
                        dfScores['Ranking'] = dfScores['Time'].rank(ascending=1)

                        # Set the index to newly created column, Rating_Rank
                        dfScores = dfScores.set_index('Ranking')
                        dff = dfScores.sort_index()
                        print(dff)

                        tabelatempos = {
                            'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
                                      'Random Forest Tree Classification', 'Perceptron Classification',
                                      'K-Means Clustering'],
                            'Time': [0.083743, 0.007015, 0.020000, 0.539887, 0.054854, 0.124665], }

                        dftempo = pd.DataFrame(tabelatempos)
                        sns.barplot(x=dftempo['Model'], y=dftempo['Time'], data=pd.melt(dftempo))
                        plt.title("Execution time of the different models")
                        plt.show()
                    else:
                        print("Invalid Option")

                    print("\n")
                    submenu4()
                    option = int(input("Enter the command you want to execute:"))
            elif option == 2:
                submenu5()
    elif option == 4:
      print("\n")
    elif option == 5:
      print("\n")

    else:
        print("Invalid Option")

    print("\n")
    menu()
    option = int(input("\nEnter the command you want to execute:"))

print("\nThank you for using this program. See you soon")