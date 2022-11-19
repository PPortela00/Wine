import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import cluster
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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
                        X = wines_binary.iloc[:, 0:12].values
                        y = wines_binary.iloc[:, 12:13].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                        print("\nShape of X_train: ", X_train.shape)
                        print("Shape of X_test: ", X_test.shape)
                        print("Shape of y_train: ", y_train.shape)
                        print("Shape of y_test", y_test.shape)

                        # Feature Scaling
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)

                        print("\nNaiveBayes")
                        classifier_nb = GaussianNB()
                        classifier_nb1 = classifier_nb.fit(X_train_scaled, y_train.ravel())
                        y_pred = classifier_nb.fit(X_train, y_train).predict(X_test)

                        cf_matrix = confusion_matrix(y_test, y_pred)
                        print(cf_matrix)

                        print("\nNumber of well predicted points out of a total %d points: %d" % (
                            X_test.shape[0], (y_test == y_pred).sum()))

                        accuracy = ((y_test == y_pred).sum() / X_test.shape[0]) * 100
                        print("Precision = {:.2f} %".format(accuracy))

                        # Predicting Cross Validation Score
                        cv_nb = cross_val_score(estimator=classifier_nb1, X=X_train_scaled, y=y_train.ravel(), cv=10)
                        print("\nCV: = {:.2f} ".format(cv_nb.mean()))

                        y_pred_nb_train = classifier_nb1.predict(X_train_scaled)
                        accuracy_nb_train = accuracy_score(y_train, y_pred_nb_train)
                        print("Training Set Accuracy:  = {:.2f} ".format(accuracy_nb_train))

                        y_pred_nb_test = classifier_nb1.predict(X_test_scaled)
                        accuracy_nb_test = accuracy_score(y_test, y_pred_nb_test)
                        print("Test Set Accuracy:  = {:.2f} ".format(accuracy_nb_test))

                        tp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 0]
                        fp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 1]
                        tn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 1]
                        fn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 0]

                    elif option == 2:
                        # Train_test split
                        X = wines_binary_norm.iloc[:, 0:12].values
                        y = wines_binary_norm.iloc[:, 12:13].values.ravel()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                        # Create a stratified 10-fold cross validation set
                        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

                        accd = dict()
                        pred = dict()
                        recd = dict()
                        f1d = dict()
                        acc = np.zeros((100))
                        pre = np.zeros((100))
                        rec = np.zeros((100))
                        f1 = np.zeros((100))
                        j = 0
                        for k in range(1, 200, 2):
                            i = 0
                            for train_index, val_index in strtfdKFold.split(X_train, y_train):
                                X_t, X_val = X_train[train_index], X_train[val_index]
                                y_t, y_val = y_train[train_index], y_train[val_index]
                                classifier_knn = KNeighborsClassifier(n_neighbors=k)
                                classifier_knn.fit(X_t, y_t)
                                yhat = classifier_knn.predict(X_val)
                                accd[i] = metrics.accuracy_score(yhat, y_val)
                                pred[i] = metrics.precision_score(yhat, y_val)
                                recd[i] = metrics.recall_score(yhat, y_val)
                                f1d[i] = metrics.f1_score(yhat, y_val)
                                i = i + 1
                            accd_test = max(accd.values())
                            i = list(accd.keys())[list(accd.values()).index(accd_test)]
                            acc[j] = accd[i]
                            pre[j] = pred[i]
                            rec[j] = recd[i]
                            f1[j] = recd[i]
                            j += 1

                        print('Mean accuracy: ' + str(np.mean(acc, axis=0)))
                        print("Mean precision: " + str(np.mean(pre, axis=0)))
                        print("Mean recall: " + str(np.mean(rec, axis=0)))
                        print("Mean f1-score: " + str(np.mean(f1, axis=0)))

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
                        X = wines_binary.iloc[:, 0:12].values
                        y = wines_binary.iloc[:, 12:13].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                        print("\nShape of X_train: ", X_train.shape)
                        print("Shape of X_test: ", X_test.shape)
                        print("Shape of y_train: ", y_train.shape)
                        print("Shape of y_test", y_test.shape)

                        # Feature Scaling
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)

                        print("\nDecision tree")
                        clf = tree.DecisionTreeClassifier()
                        clf = clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        print(confusion_matrix(y_test, y_pred))

                        print("\nNumber of well predicted points out of a total %d points: %d" % (
                            X_test.shape[0], (y_test == y_pred).sum()))
                        accuracy = ((y_test == y_pred).sum() / X_test.shape[0]) * 100
                        print("Precision = {:.2f} %".format(accuracy))

                        # Fitting classifier to the Training set
                        classifier_dt = DecisionTreeClassifier(criterion='gini', max_features=6, max_leaf_nodes=400,
                                                               random_state=33)
                        classifier_dt.fit(X_train_scaled, y_train.ravel())

                        # Predicting Cross Validation Score
                        cv_dt = cross_val_score(estimator=classifier_dt, X=X_train_scaled, y=y_train.ravel(), cv=10)
                        print("\nCV:  = {:.2f} ".format(cv_dt.mean()))

                        y_pred_dt_train = classifier_dt.predict(X_train_scaled)
                        accuracy_dt_train = accuracy_score(y_train, y_pred_dt_train)
                        print("Training Set Accuracy:  = {:.2f} ".format(accuracy_dt_train))

                        y_pred_dt_test = classifier_dt.predict(X_test_scaled)
                        accuracy_dt_test = accuracy_score(y_test, y_pred_dt_test)
                        print("Test Set Accuracy:  = {:.2f} ".format(accuracy_dt_test))

                        confusion_matrix(y_test, y_pred_dt_test)

                        tp_dt = confusion_matrix(y_test, y_pred_dt_test)[0, 0]
                        fp_dt = confusion_matrix(y_test, y_pred_dt_test)[0, 1]
                        tn_dt = confusion_matrix(y_test, y_pred_dt_test)[1, 1]
                        fn_dt = confusion_matrix(y_test, y_pred_dt_test)[1, 0]

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
                        X = wines_binary.iloc[:, 0:12].values
                        y = wines_binary.iloc[:, 12:13].values.ravel()

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                        print("\nShape of X_train: ", X_train.shape)
                        print("Shape of X_test: ", X_test.shape)
                        print("Shape of y_train: ", y_train.shape)
                        print("Shape of y_test", y_test.shape)

                        # Feature Scaling
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)

                        print("\nRandom Forest Classification")
                        rf = RandomForestClassifier(max_depth=10, random_state=0)
                        clf = rf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        print(confusion_matrix(y_test, y_pred))

                        print("\nNumber of well predicted points out of a total %d points: %d" % (
                            X_test.shape[0], (y_test == y_pred).sum()))
                        accuracy = ((y_test == y_pred).sum() / X_test.shape[0]) * 100
                        print("Precision = {:.2f} %".format(accuracy))

                        classifier_rf = RandomForestClassifier(criterion='entropy', max_features=4, n_estimators=800,
                                                               random_state=33)
                        classifier_rf.fit(X_train_scaled, y_train.ravel())

                        # Predicting Cross Validation Score
                        cv_rf = cross_val_score(estimator=classifier_rf, X=X_train_scaled, y=y_train.ravel(), cv=10)
                        print("CV:  = {:.2f} ".format(cv_rf.mean()))

                        y_pred_rf_train = classifier_rf.predict(X_train_scaled)
                        accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
                        print("Training Set Accuracy:  = {:.2f} ".format(accuracy_rf_train))

                        y_pred_rf_test = classifier_rf.predict(X_test_scaled)
                        accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
                        print("Test Set Accuracy:  = {:.2f} ".format(accuracy_rf_test))

                        tp_rf = confusion_matrix(y_test, y_pred_rf_test)[0, 0]
                        fp_rf = confusion_matrix(y_test, y_pred_rf_test)[0, 1]
                        tn_rf = confusion_matrix(y_test, y_pred_rf_test)[1, 1]
                        fn_rf = confusion_matrix(y_test, y_pred_rf_test)[1, 0]

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