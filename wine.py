import warnings

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
#%matplotlib inline

from scipy import interp
from sklearn import metrics
from sklearn.preprocessing import label_binarize, StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from mlxtend.classifier import StackingClassifier
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
    print("[1] NaiveBayes - Smote")
    print("[1] NaiveBayes - Oversampling")
    print("[2] KNN - Smote")
    print("[2] KNN - Oversampling")
    print("[3] Logistic Regression - Smote")
    print("[3] Logistic Regression - Oversampling")
    print("[4] Decision Tree - Smote")
    print("[4] Decision Tree - Oversampling")
    print("[5] Model Tree - Smote")
    print("[5] Model Tree - Oversampling")
    print("[6] Artificial Neural Networks - Smote")
    print("[6] Artificial Neural Networks - Oversampling")
    print("[7] Support Vector Machine - Smote")
    print("[7] Support Vector Machine - Oversampling")
    print("[8] Random Forest - Smote")
    print("[8] Random Forest - Oversampling")
    print("\n[0] Return to the program's main menu")

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
                    #plots = interpreter.feature_importance.plot_feature_importance(im_model.feature_names, progressbar=False,
                    #                                                              n_jobs=1, ascending=True)

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

                class_ql = {'low': 0, 'medium': 1, 'high': 2}
                y_ql = wines_binary.quality.map(class_ql)

                wqp_class_labels = np.array(wines_binary['quality'])
                target_names = ['low', 'medium', 'high']

                cols = wines_binary.columns
                cols = list(cols.drop(['quality']))
                X_train, X_test, y_train, y_test = train_test_split(wines_binary.loc[:, cols], y_ql.values, test_size=0.20,
                                                                random_state=101)


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


                cols_clean = cols.copy()
                cols_clean.remove('total_sulfur_dioxide')
                cols_clean.remove('residual_sugar')

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
                option = int(input("\nInsert the command that you want to execute:\n"))
                while option != 0:
                    if option == 1:
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
                        clf = Pipeline([
                            # ('pca', PCA(random_state = 101)),
                            ('clf', LogisticRegression(random_state=101))])

                        # a list of dictionaries to specify the parameters that we'd want to tune
                        SEL = cols_clean
                        n_components = [len(SEL) - 2, len(SEL) - 1, len(SEL)]
                        whiten = [True, False]
                        C = [1.0]  # , 1e-06, 5e-07, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 10.0, 100.0, 1000.0]
                        tol = [1e-06]  # , 5e-07, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]

                        param_grid = \
                            [{'clf__C': C
                                 , 'clf__solver': ['liblinear', 'saga']
                                 , 'clf__penalty': ['l1', 'l2']
                                 , 'clf__tol': tol
                                 , 'clf__class_weight': ['balanced']
                              # ,'pca__n_components' : n_components
                              # ,'pca__whiten' : whiten
                              },
                             {'clf__C': C
                                 , 'clf__max_iter': [3, 9, 2, 7, 4]
                                 , 'clf__solver': ['newton-cg', 'sag', 'lbfgs']
                                 , 'clf__penalty': ['l2']
                                 , 'clf__tol': tol
                                 , 'clf__class_weight': ['balanced']
                              # ,'pca__n_components' : n_components
                              # ,'pca__whiten' : whiten
                              }]

                        gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1,
                                          n_jobs=-1)

                        LR = Pipeline([
                            ('sel', select_fetaures(select_cols=SEL)),
                            ('scl', StandardScaler()),
                            ('gs', gs)
                        ])

                        LR.fit(X_train, y_train)

                        results = get_results(LR, 'LogisticRegression', X_test, y_test,
                                              target_names=target_names, results=results, reasume=False)

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
                        print("\n")

                    elif option == 6:
                        print("\n")

                    elif option == 7:
                        print("\n")

                    elif option == 8:
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

                        print(gs.best_params_)
                        print(gs.best_score_)

                        results = get_results(RF, 'RF', X_test, y_test,
                                              target_names=target_names, results=None, reasume=True)

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