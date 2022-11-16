import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from tabulate import tabulate
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from timeit import timeit
from sklearn import datasets, tree


pd.set_option('display.max_columns', None)                  #para poder visualizar todas as colunas no display
pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', None)                  #para poder visualizar todas as colunas no display
pd.set_option('display.width', 1000)                        # para a largura do display ser de dimensao 1000

df_red = pd.read_csv("redwine.csv",delimiter=";")
df_white = pd.read_csv("whitewine.csv",delimiter=";")
df_red.columns = df_red.columns.str.replace(' ', '_')       # torna mais facil a utilizaçao das colunas
df_white.columns = df_white.columns.str.replace(' ', '_')   # torna mais facil a utilizaçao das colunas

df_red["color"] = "R"
df_white["color"] = "W"
df_all = pd.concat([df_red, df_white], axis=0)
df_all.columns = df_all.columns.str.replace(' ', '_')       # torna mais facil a utilizaçao das colunas
df_all['color'] = df_all['color'].apply(lambda x: 0 if x == 'W' else 1)

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
    print("[1] Print to all existing wines")
    print("[2] Preparing data for the dataset")
    print("[3] Check how many null values exist")
    print("[4] Create descriptive statistics (only for numeric columns)")
    print("[5] Histogram showing all components of white and red wine")
    print("[6] Distribution of red and white wine taking into account the quality")
    print("[7] Existing correlation between quality and the various attributes")
    print("[8] HeatMaps - display correlations between quality and various attributes for the white and red wine")
    print("[9] HeatMap - displays the correlation difference between red wine and white wine")
    print("[10] Existing covariance between quality and the various attributes")
    print("[11] Red wine stripplots - characteristics for which it had the highest correlation")
    print("[12] White wine stripplots - characteristics for which it had the highest correlation")
    print("[13] Red Wine Regplots - express interesting correlations between different components")
    print("[14] White Wine Regplots - express interesting correlations between different components")
    print("[15] ScatterPlots")

    print("\n[0] Return to the program's main menu")


def submenu2():
    print("\n")
    print("------ Data Preparation Menu ------")
    print("[1] Normalization")
    print("[2] Standardization")

    print("\n[0] Return to the program's main menu")


def submenu3():
    print("\n")
    print("------ Modeling (ML Algorithms Application) Menu ------")
    print("[1] NaiveBayes")
    print("[2] Knn")
    print("[3] Better k for the KNN")
    print("[4] Cross-Validation and Training and Test Set - Red Wine")
    print("[5] Cross-Validation and Training and Test Set - White Wine")
    print("[6] Decision Tree - Classification")
    print("[7] Random forests classification")
    print("[8] Confusion Matrix")
    print("[9] Accuracy models - White Wine")
    print("[10] Accuracy models - Red Wine")
    print("[11] Tabela - White Wine")
    print("[12] Tabela - Red Wine")
    print("[13] K-Means Clustering - White Wine")
    print("[14] K-Means Clustering - Red Wine")
    print("[15] Hierarchical Clustering")
    print("[16] Perceptron Classification - White Wine")
    print("[17] Perceptron Classification - Red Wine")
    print("[18] Time - White Wine")
    print("[19] Time - Red Wine")

    print("[0] Return to the program's main menu")


while option != 0:
    if option == 1:
        submenu1()
        option = int(input("\nInsert the command that you want to execute:\n"))

        while option != 0:
            if option == 1:
                print('\n')
                print(df_all.head())

            elif option == 2:
                print('\n')
                print('Number of matrix elements with both wines')
                print(df_all.size)
                print('\n')
                print('Matrix dimension with both wines')
                print(df_all.shape)
                print('\n')
                print("Total number of data in each attribute")
                print(df_all.count())
                print('\n')
                print("Type of variables and amount of data in each column")
                df_all.info()

            elif option == 3:
                nulos= df_all.isnull().sum()
                print('\n')
                print(nulos)
                print('\n')
                print("There are no null values in this dataset!!!")

            elif option == 4:
                print("\n")
                print("Red Wine")
                print(df_red.describe())
                print("\n")
                print("White Wine")
                print(df_red.describe())
                print("\n")
                print("Both wines")
                print(df_all.describe())
                print("\n")

                d = {'Cor': ['Vinho Tinto', 'Vinho Branco'], 'Qualidade_media': [df_red["quality"].mean(), df_white["quality"].mean()], 'Qualidade_Maxima': [df_red["quality"].max(), df_white["quality"].max()],'Qualidade_Minima': [df_red["quality"].min(), df_white["quality"].min()]}
                df = pd.DataFrame(data=d)
                print(df)

            elif option == 5:
                df_red.hist()
                plt.tight_layout(pad=1.1)
                plt.suptitle('Red Wine - distribution of the various components', fontsize=13)
                df_white.hist()
                plt.tight_layout(pad=1.1)
                plt.suptitle('White Wine - distribution of the various components', fontsize=13)
                plt.show()

            elif option == 6:
                sns.violinplot(x="quality", data=df_red)
                plt.suptitle('Distribution of Red Wine taking into account the quality', fontsize=12)
                plt.figure()
                sns.violinplot(x="quality", data=df_white)
                plt.suptitle('Distribution of White Wine taking into account the quality', fontsize=12)
                plt.show()

            elif option == 7:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                correlationred = df_red.select_dtypes(include=numerics).corr()['quality'].sort_values(ascending=False)
                correlationwhite = df_white.select_dtypes(include=numerics).corr()['quality'].sort_values(ascending=False)
                print("\n")
                print('Red Wine')
                print(correlationred)
                print("\n")
                print('White Wine')
                print(correlationwhite)

            elif option == 8:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                corr = df_red.select_dtypes(include=numerics).corr()
                print('\n')
                print('Correlations between the various attributes of red wine')
                print(corr)
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                            cmap=sns.diverging_palette(220, 20, as_cmap=True))

                plt.suptitle('HeatMap- Red Wine', fontsize=13)
                plt.tight_layout()
                corr1 = df_white.select_dtypes(include=numerics).corr()
                print('\n')
                print('Correlations between the various attributes of wjite wine')
                print(corr1)
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr1, xticklabels=corr1.columns, yticklabels=corr1.columns, annot=True,
                            cmap=sns.diverging_palette(220, 20, as_cmap=True))

                plt.suptitle('HeatMap- White Wine', fontsize=13)
                plt.tight_layout()
                plt.show()

            elif option == 9:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                dif_corr = df_red.select_dtypes(include=numerics).corr() - df_white.select_dtypes(include=numerics).corr()

                plt.subplots(figsize=(12, 6))
                sns.heatmap(dif_corr, xticklabels=dif_corr.columns, yticklabels=dif_corr.columns, annot=True,
                            cmap=sns.diverging_palette(220, 20, as_cmap=True))

                plt.suptitle('HeatMap- Correlation difference between red and white wine', fontsize=15)
                plt.tight_layout()
                plt.show()

            elif option == 10:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                covarianciared = df_red.select_dtypes(include=numerics).cov()['quality'].sort_values(ascending=False)
                covarianciawhite = df_white.select_dtypes(include=numerics).cov()['quality'].sort_values(ascending=False)

                print("\n")
                print('Red Wine')
                print(covarianciared)
                print("\n")
                print('White Wine')
                print(covarianciawhite)

                corr = df_red.select_dtypes(include=numerics).cov()
                print('\n')
                print('Covariances between the various attributes of red wine')
                print(corr)
                plt.subplots(figsize=(12, 6))
                sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                        cmap=sns.diverging_palette(220, 20, as_cmap=True))

                plt.suptitle('Covariances between the various attributes of red wine', fontsize=15)
                plt.tight_layout()

                corr1 = df_white.select_dtypes(include=numerics).cov()
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
                sns.boxplot(orient='v', data=df_red, y="alcohol", x="quality")
                sns.stripplot(data=df_red, y="alcohol", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & sulfates', fontsize=15)
                sns.boxplot(orient='v', data=df_red, y="sulphates", x="quality")
                sns.stripplot(data=df_red, y="sulphates", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & volatile acidity ', fontsize=15)
                sns.boxplot(orient='v', data=df_red, y="volatile_acidity", x="quality")
                sns.stripplot(data=df_red, y="volatile_acidity", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('Red Wine - quality & citric acid', fontsize=15)
                sns.boxplot(orient='v', data=df_red, y="citric_acid", x="quality")
                sns.stripplot(data=df_red, y="citric_acid", x="quality")
                plt.show()

            elif option == 12:

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & alcohol', fontsize=15)
                sns.boxplot(orient='v', data=df_white, y="alcohol", x="quality")
                sns.stripplot(data=df_white, y="alcohol", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & sulfates', fontsize=15)
                sns.boxplot(orient='v', data=df_white, y="sulphates", x="quality")
                sns.stripplot(data=df_white, y="sulphates", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & pH', fontsize=15)
                sns.boxplot(orient='v', data=df_white, y="pH", x="quality")
                sns.stripplot(data=df_white, y="pH", x="quality")

                plt.figure(figsize=(6, 4))
                plt.suptitle('White wine - quality & density', fontsize=15)
                sns.boxplot(orient='v', data=df_white, y="density", x="quality")
                sns.stripplot(data=df_white, y="density", x="quality")

                plt.show()

            elif option == 13:
                fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
                sns.regplot(x="alcohol", y="density", data=df_red, ax=ax1, scatter_kws={'s': 2})
                sns.regplot(x="fixed_acidity", y="density", data=df_red, ax=ax2, scatter_kws={'s': 2})
                sns.regplot(x="citric_acid", y="fixed_acidity", data=df_red, ax=ax3, scatter_kws={'s': 2})
                sns.regplot(x="pH", y="fixed_acidity", data=df_red, ax=ax4, scatter_kws={'s': 2})
                plt.suptitle('Red Wine - Interesting Correlations', fontsize=15)
                plt.show()

            elif option == 14:
                fig, ((ax1, ax2) , (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

                sns.regplot(x="alcohol", y="density", data=df_white,ax=ax1, scatter_kws={'s': 2})
                sns.regplot(x="total_sulfur_dioxide", y="density", data=df_white, ax=ax2, scatter_kws={'s': 2})
                sns.regplot(x="density", y="residual_sugar", data=df_white, ax=ax3, scatter_kws={'s': 2})
                sns.regplot(x="alcohol", y="residual_sugar", data=df_white, ax=ax4, scatter_kws={'s': 2})
                plt.suptitle('White Wine - Interesting Correlations', fontsize=15)
                plt.show()

            elif option == 15:
                sns.scatterplot(x='free_sulfur_dioxide', y='total_sulfur_dioxide', hue='color', data=df_all)
                plt.suptitle('All Wines - Free Sulfur Dioxide vs Total Sulfur Dioxide', fontsize=15)
                plt.figure()
                sns.scatterplot(x='residual_sugar', y='density', hue='color', data=df_all)
                plt.suptitle('All Wines - Residual Sugar vs Density', fontsize=15)
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
                print('\n')
                print(df_all.head(5))

                # normalizaçao
                print('\n')
                scaler = MinMaxScaler()
                scaler.fit(df_all)
                scaled_features = scaler.transform(df_all)
                print(scaled_features)

                # volta a converter em dataframe
                print('\n')
                new = pd.DataFrame(data=scaled_features,
                                   columns=["fixed_acidity",
                                            "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                            "free_sulfur_dioxide",
                                            "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality",
                                            "color"])
                print(new)
                print('\n')
                print(type(scaled_features))

                # graficos antes e depois da normalizaçao
                sns.displot(new['quality']).set(title='Normalized')
                sns.displot(df_all['quality']).set(title='Not Normalized')
                plt.show()

            elif option == 2:
                print('\n')
                print(df_all.head(5))

                print('\n')
                from sklearn.preprocessing import StandardScaler

                sc_X = StandardScaler()
                sc_X = sc_X.fit_transform(df_all)
                print(sc_X)

                print('\n')
                new = pd.DataFrame(data=sc_X,
                                   columns=["fixed_acidity",
                                            "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                            "free_sulfur_dioxide",
                                            "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality",
                                            "color"])
                print(new)
                print('\n')
                print(type(sc_X))

                sns.displot(new['sulphates']).set(title='Standardized')
                sns.displot(df_all['quality']).set(title='Not Standardized')
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
                print("\nRed Wine")

                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()

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

                print("\nWhite Wine")
                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                print("NaiveBayes")
                gnb = GaussianNB()
                y_pred = gnb.fit(X_train, y_train).predict(X_test)
                print(confusion_matrix(y_test, y_pred))
                print("Number of mislabeled points out of a total %d points : %d" % (
                X_test.shape[0], (y_test != y_pred).sum()))

                accuracy1 = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                print(accuracy1, "%")

                models = [
                    ('K-Nearest Neighbors (KNN) - Red Wine', accuracy),
                    ('K-Nearest Neighbors (KNN) - White Wine', accuracy1)]

                predict = pd.DataFrame(data=models, columns=['Model', 'KNN'])
                print(predict)
            elif option == 2:
                print("\nRed Wine")

                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                neigh = KNeighborsClassifier(n_neighbors=25)
                y_pred = neigh.fit(X_train, y_train).predict(X_test)

                print(confusion_matrix(y_test, y_pred))
                print("Number of mislabeled points out of a total %d points : %d" % (
                X_test.shape[0], (y_test != y_pred).sum()))

                accuracy = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                print(accuracy, "%")

                print("\nWhite Wine")
                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                neigh = KNeighborsClassifier(n_neighbors=48)
                y_pred = neigh.fit(X_train, y_train).predict(X_test)
                print(confusion_matrix(y_test, y_pred))
                print("Number of mislabeled points out of a total %d points : %d" % (
                X_test.shape[0], (y_test != y_pred).sum()))

                accuracy = ((y_test != y_pred).sum() / X_test.shape[0]) * 100
                print(accuracy, "%")

            elif option == 3:
                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                print("\nTrain set:", X_train.shape, y_train.shape)
                print("Test set:", X_test.shape, y_test.shape)

                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.model_selection import cross_val_score

                # Number of k from 1 to 50
                k_range = range(1, 50)
                k_scores = []
                # Calculate cross validation score for every k number from 1 to 50
                for k in k_range:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    # It’s 5 fold cross validation with ‘accuracy’ scoring
                    scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
                    k_scores.append(scores.mean())

                # Plot accuracy for every k number between 1 and 50
                plt.plot(k_range, k_scores)
                plt.suptitle("Best k for KNN - Red Wine")
                plt.xlabel("Value of K for KNN - Red Wine")
                plt.ylabel("Cross-validated accuracy")
                plt.figure()

                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                print("\nTrain set:", X_train.shape, y_train.shape)
                print("Test set:", X_test.shape, y_test.shape)

                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.model_selection import cross_val_score

                # Number of k from 1 to 50
                k_range = range(1, 50)
                k_scores = []
                # Calculate cross validation score for every k number from 1 to 50
                for k in k_range:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    # It’s 5 fold cross validation with ‘accuracy’ scoring
                    scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
                    k_scores.append(scores.mean())

                # Plot accuracy for every k number between 1 and 50
                plt.plot(k_range, k_scores)
                plt.suptitle("Best k for KNN - White Wine")
                plt.xlabel("Value of K for KNN - White Wine")
                plt.ylabel("Cross-validated accuracy")
                plt.show()
            elif option == 4:

                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                print("Shape of X_train: ", X_train.shape)
                print("Shape of X_test: ", X_test.shape)
                print("Shape of y_train: ", y_train.shape)
                print("Shape of y_test", y_test.shape)

                from sklearn.preprocessing import StandardScaler

                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(X_train)
                X_test_scaled = sc.transform(X_test)

                from sklearn.neighbors import KNeighborsClassifier

                classifier_knn = KNeighborsClassifier(leaf_size=1, metric='minkowski', n_neighbors=32,
                                                      weights='distance')
                classifier_knn.fit(X_train_scaled, y_train.ravel())

                print("\n")
                print("KNN")
                # Predicting Cross Validation Score
                cv_knn = cross_val_score(estimator=classifier_knn, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_knn.mean())

                y_pred_knn_train = classifier_knn.predict(X_train_scaled)
                accuracy_knn_train = accuracy_score(y_train, y_pred_knn_train)
                print("Training set: ", accuracy_knn_train)

                y_pred_knn_test = classifier_knn.predict(X_test_scaled)
                accuracy_knn_test = accuracy_score(y_test, y_pred_knn_test)
                print("Test set: ", accuracy_knn_test)

                print(confusion_matrix(y_test, y_pred_knn_test))

                tp_knn = confusion_matrix(y_test, y_pred_knn_test)[0, 0]
                fp_knn = confusion_matrix(y_test, y_pred_knn_test)[0, 1]
                tn_knn = confusion_matrix(y_test, y_pred_knn_test)[1, 1]
                fn_knn = confusion_matrix(y_test, y_pred_knn_test)[1, 0]

                # Fitting classifier to the Training set
                from sklearn.naive_bayes import GaussianNB

                classifier_nb = GaussianNB()
                classifier_nb.fit(X_train_scaled, y_train.ravel())

                print("\n")
                print("NaiveBayes")
                # Predicting Cross Validation Score
                cv_nb = cross_val_score(estimator=classifier_nb, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_nb.mean())

                y_pred_nb_train = classifier_nb.predict(X_train_scaled)
                accuracy_nb_train = accuracy_score(y_train, y_pred_nb_train)
                print("Training set: ", accuracy_nb_train)

                y_pred_nb_test = classifier_nb.predict(X_test_scaled)
                accuracy_nb_test = accuracy_score(y_test, y_pred_nb_test)
                print("Test set: ", accuracy_nb_test)

                print(confusion_matrix(y_test, y_pred_nb_test))

                tp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 0]
                fp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 1]
                tn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 1]
                fn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 0]

                print("\n")
                print("Decision Tree")
                from sklearn.tree import DecisionTreeClassifier, export_graphviz

                classifier_dt = DecisionTreeClassifier(criterion='gini', max_features=6, max_leaf_nodes=400,
                                                       random_state=33)
                classifier_dt.fit(X_train_scaled, y_train.ravel())
                # Predicting Cross Validation Score
                cv_dt = cross_val_score(estimator=classifier_dt, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_dt.mean())

                y_pred_dt_train = classifier_dt.predict(X_train_scaled)
                accuracy_dt_train = accuracy_score(y_train, y_pred_dt_train)
                print("Training set: ", accuracy_dt_train)

                y_pred_dt_test = classifier_dt.predict(X_test_scaled)
                accuracy_dt_test = accuracy_score(y_test, y_pred_dt_test)
                print("Test set: ", accuracy_dt_test)

                confusion_matrix(y_test, y_pred_dt_test)

                tp_dt = confusion_matrix(y_test, y_pred_dt_test)[0, 0]
                fp_dt = confusion_matrix(y_test, y_pred_dt_test)[0, 1]
                tn_dt = confusion_matrix(y_test, y_pred_dt_test)[1, 1]
                fn_dt = confusion_matrix(y_test, y_pred_dt_test)[1, 0]

                print("\n")
                print("Random Forest Classification")
                from sklearn.ensemble import RandomForestClassifier

                classifier_rf = RandomForestClassifier(criterion='entropy', max_features=4, n_estimators=800,
                                                       random_state=33)
                classifier_rf.fit(X_train_scaled, y_train.ravel())

                # Predicting Cross Validation Score
                cv_rf = cross_val_score(estimator=classifier_rf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_rf.mean())

                y_pred_rf_train = classifier_rf.predict(X_train_scaled)
                accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
                print("Training set: ", accuracy_rf_train)

                y_pred_rf_test = classifier_rf.predict(X_test_scaled)
                accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
                print("Test set: ", accuracy_rf_test)

                confusion_matrix(y_test, y_pred_rf_test)

                tp_rf = confusion_matrix(y_test, y_pred_rf_test)[0, 0]
                fp_rf = confusion_matrix(y_test, y_pred_rf_test)[0, 1]
                tn_rf = confusion_matrix(y_test, y_pred_rf_test)[1, 1]
                fn_rf = confusion_matrix(y_test, y_pred_rf_test)[1, 0]

                print("\n")
                print("Perceptron Classification")

                clf = Perceptron(verbose=3)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Predicting Cross Validation Score
                cv_pc = cross_val_score(estimator=clf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_pc.mean())

                y_pred_pc_train = clf.predict(X_train_scaled)
                accuracy_pc_train = accuracy_score(y_train, y_pred_pc_train)
                print("Training set: ", accuracy_rf_train)

                y_pred_pc_test = classifier_rf.predict(X_test_scaled)
                accuracy_pc_test = accuracy_score(y_test, y_pred_pc_test)
                print("Test set: ", accuracy_pc_test)

                confusion_matrix(y_test, y_pred_pc_test)

                tp_pc = confusion_matrix(y_test, y_pred_pc_test)[0, 0]
                fp_pc = confusion_matrix(y_test, y_pred_pc_test)[0, 1]
                tn_pc = confusion_matrix(y_test, y_pred_pc_test)[1, 1]
                fn_pc = confusion_matrix(y_test, y_pred_pc_test)[1, 0]

                models = [
                    ('K-Nearest Neighbors (KNN)', tp_knn, fp_knn, tn_knn, fn_knn, accuracy_knn_train, accuracy_knn_test,
                     cv_knn.mean()),
                    ('Naive Bayes', tp_nb, fp_nb, tn_nb, fn_nb, accuracy_nb_train, accuracy_nb_test, cv_nb.mean()),
                    ('Decision Tree Classification', tp_dt, fp_dt, tn_dt, fn_dt, accuracy_dt_train, accuracy_dt_test,
                     cv_dt.mean()),
                    ('Random Forest Tree Classification', tp_rf, fp_rf, tn_rf, fn_rf, accuracy_rf_train,
                     accuracy_rf_test, cv_rf.mean()),
                    ('Perceptron Classification', tp_pc, fp_pc, tn_pc, fn_pc, accuracy_pc_train, accuracy_pc_test,
                     cv_pc.mean())]

                predict = pd.DataFrame(data=models,
                                       columns=['Model', 'True Positive', 'False Positive', 'True Negative',
                                                'False Negative', 'Precision(training)', 'Precision(test)',
                                                'Cross-Validation'])
                print(predict)

                f, axe = plt.subplots(1, 1, figsize=(18, 6))

                predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

                sns.barplot(x='Cross-Validation', y='Model', data=predict, ax=axe)
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_title('Cross-Validation Scores for the Models - Red Wine')
                axe.set_xlabel('Cross-Validaton Score', size=16)
                axe.set_ylabel('Model')
                axe.set_xlim(0, 1.0)
                axe.set_xticks(np.arange(0, 1.1, 0.1))
                plt.show()

                f, axes = plt.subplots(2, 1, figsize=(14, 10))

                predict.sort_values(by=['Precision(training)'], ascending=False, inplace=True)

                sns.barplot(x='Precision(training)', y='Model', data=predict, palette='Blues_d', ax=axes[0])
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_title('Training Precisions for the Models - Red Wine')
                axes[0].set_xlabel('Precision (Training)', size=16)
                axes[0].set_ylabel('Model')
                axes[0].set_xlim(0, 1.0)
                axes[0].set_xticks(np.arange(0, 1.1, 0.1))

                print("\n")

                predict.sort_values(by=['Precision(test)'], ascending=False, inplace=True)

                sns.barplot(x='Precision(test)', y='Model', data=predict, palette='Reds_d', ax=axes[1])
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_title('Test Precisions for the Models - Red Wine')
                axes[1].set_xlabel('Precision (Test)', size=16)
                axes[1].set_ylabel('Model')
                axes[1].set_xlim(0, 1.0)
                axes[1].set_xticks(np.arange(0, 1.1, 0.1))

                plt.show()

            elif option == 5:
                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                print("Shape of X_train: ", X_train.shape)
                print("Shape of X_test: ", X_test.shape)
                print("Shape of y_train: ", y_train.shape)
                print("Shape of y_test", y_test.shape)

                from sklearn.preprocessing import StandardScaler

                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(X_train)
                X_test_scaled = sc.transform(X_test)

                from sklearn.neighbors import KNeighborsClassifier

                classifier_knn = KNeighborsClassifier(leaf_size=1, metric='minkowski', n_neighbors=32,
                                                      weights='distance')
                classifier_knn.fit(X_train_scaled, y_train.ravel())

                print("\n")
                print("KNN")
                # Predicting Cross Validation Score
                cv_knn = cross_val_score(estimator=classifier_knn, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_knn.mean())

                y_pred_knn_train = classifier_knn.predict(X_train_scaled)
                accuracy_knn_train = accuracy_score(y_train, y_pred_knn_train)
                print("Training set: ", accuracy_knn_train)

                y_pred_knn_test = classifier_knn.predict(X_test_scaled)
                accuracy_knn_test = accuracy_score(y_test, y_pred_knn_test)
                print("Test set: ", accuracy_knn_test)

                print(confusion_matrix(y_test, y_pred_knn_test))

                tp_knn = confusion_matrix(y_test, y_pred_knn_test)[0, 0]
                fp_knn = confusion_matrix(y_test, y_pred_knn_test)[0, 1]
                tn_knn = confusion_matrix(y_test, y_pred_knn_test)[1, 1]
                fn_knn = confusion_matrix(y_test, y_pred_knn_test)[1, 0]

                # Fitting classifier to the Training set
                from sklearn.naive_bayes import GaussianNB

                classifier_nb = GaussianNB()
                classifier_nb.fit(X_train_scaled, y_train.ravel())

                print("\n")
                print("NaiveBayes")
                # Predicting Cross Validation Score
                cv_nb = cross_val_score(estimator=classifier_nb, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_nb.mean())

                y_pred_nb_train = classifier_nb.predict(X_train_scaled)
                accuracy_nb_train = accuracy_score(y_train, y_pred_nb_train)
                print("Training set: ", accuracy_nb_train)

                y_pred_nb_test = classifier_nb.predict(X_test_scaled)
                accuracy_nb_test = accuracy_score(y_test, y_pred_nb_test)
                print("Test set: ", accuracy_nb_test)

                print(confusion_matrix(y_test, y_pred_nb_test))

                tp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 0]
                fp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 1]
                tn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 1]
                fn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 0]

                print("\n")
                print("Decision Tree")
                from sklearn.tree import DecisionTreeClassifier

                classifier_dt = DecisionTreeClassifier(criterion='gini', max_features=6, max_leaf_nodes=400,
                                                       random_state=33)
                classifier_dt.fit(X_train_scaled, y_train.ravel())
                # Predicting Cross Validation Score
                cv_dt = cross_val_score(estimator=classifier_dt, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_dt.mean())

                y_pred_dt_train = classifier_dt.predict(X_train_scaled)
                accuracy_dt_train = accuracy_score(y_train, y_pred_dt_train)
                print("Training set: ", accuracy_dt_train)

                y_pred_dt_test = classifier_dt.predict(X_test_scaled)
                accuracy_dt_test = accuracy_score(y_test, y_pred_dt_test)
                print("Test set: ", accuracy_dt_test)

                confusion_matrix(y_test, y_pred_dt_test)

                tp_dt = confusion_matrix(y_test, y_pred_dt_test)[0, 0]
                fp_dt = confusion_matrix(y_test, y_pred_dt_test)[0, 1]
                tn_dt = confusion_matrix(y_test, y_pred_dt_test)[1, 1]
                fn_dt = confusion_matrix(y_test, y_pred_dt_test)[1, 0]

                print("\n")
                print("Random Forest Classification")
                from sklearn.ensemble import RandomForestClassifier

                classifier_rf = RandomForestClassifier(criterion='entropy', max_features=4, n_estimators=800,
                                                       random_state=33)
                classifier_rf.fit(X_train_scaled, y_train.ravel())

                # Predicting Cross Validation Score
                cv_rf = cross_val_score(estimator=classifier_rf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_rf.mean())

                y_pred_rf_train = classifier_rf.predict(X_train_scaled)
                accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
                print("Training set: ", accuracy_rf_train)

                y_pred_rf_test = classifier_rf.predict(X_test_scaled)
                accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
                print("Test set: ", accuracy_rf_test)

                confusion_matrix(y_test, y_pred_rf_test)

                tp_rf = confusion_matrix(y_test, y_pred_rf_test)[0, 0]
                fp_rf = confusion_matrix(y_test, y_pred_rf_test)[0, 1]
                tn_rf = confusion_matrix(y_test, y_pred_rf_test)[1, 1]
                fn_rf = confusion_matrix(y_test, y_pred_rf_test)[1, 0]

                print("\n")
                print("Perceptron Classification")

                clf = Perceptron(verbose=3)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Predicting Cross Validation Score
                cv_pc = cross_val_score(estimator=clf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV: ", cv_pc.mean())

                y_pred_pc_train = clf.predict(X_train_scaled)
                accuracy_pc_train = accuracy_score(y_train, y_pred_pc_train)
                print("Training set: ", accuracy_rf_train)

                y_pred_pc_test = classifier_rf.predict(X_test_scaled)
                accuracy_pc_test = accuracy_score(y_test, y_pred_pc_test)
                print("Test set: ", accuracy_pc_test)

                confusion_matrix(y_test, y_pred_pc_test)

                tp_pc = confusion_matrix(y_test, y_pred_pc_test)[0, 0]
                fp_pc = confusion_matrix(y_test, y_pred_pc_test)[0, 1]
                tn_pc = confusion_matrix(y_test, y_pred_pc_test)[1, 1]
                fn_pc = confusion_matrix(y_test, y_pred_pc_test)[1, 0]

                models = [
                    ('K-Nearest Neighbors (KNN)', tp_knn, fp_knn, tn_knn, fn_knn, accuracy_knn_train, accuracy_knn_test,
                     cv_knn.mean()),
                    ('Naive Bayes', tp_nb, fp_nb, tn_nb, fn_nb, accuracy_nb_train, accuracy_nb_test, cv_nb.mean()),
                    ('Decision Tree Classification', tp_dt, fp_dt, tn_dt, fn_dt, accuracy_dt_train, accuracy_dt_test,
                     cv_dt.mean()),
                    ('Random Forest Tree Classification', tp_rf, fp_rf, tn_rf, fn_rf, accuracy_rf_train,
                     accuracy_rf_test,
                     cv_rf.mean()),
                    ('Perceptron Classification ', tp_pc, fp_pc, tn_pc, fn_pc, accuracy_pc_train, accuracy_pc_test,
                     cv_pc.mean())]

                predict = pd.DataFrame(data=models,
                                       columns=['Model', 'True Positive', 'False Positive', 'True Negative',
                                                'False Negative', 'Precision(training)', 'Precision(test)',
                                                'Cross-Validation'])
                print(predict)

                f, axe = plt.subplots(1, 1, figsize=(18, 6))

                predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

                sns.barplot(x='Cross-Validation', y='Model', data=predict, ax=axe)
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_title('Cross-Validation scores for the Models - White Wine')
                axe.set_xlabel('Cross-Validaton Score', size=16)
                axe.set_ylabel('Model')
                axe.set_xlim(0, 1.0)
                axe.set_xticks(np.arange(0, 1.1, 0.1))
                plt.show()

                f, axes = plt.subplots(2, 1, figsize=(14, 10))

                predict.sort_values(by=['Precision(training)'], ascending=False, inplace=True)

                sns.barplot(x='Precision(training)', y='Model', data=predict, palette='Blues_d', ax=axes[0])
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_title('Training Precisions for the Models - White Wine')
                axes[0].set_xlabel('Precision (Training)', size=16)
                axes[0].set_ylabel('Model')
                axes[0].set_xlim(0, 1.0)
                axes[0].set_xticks(np.arange(0, 1.1, 0.1))

                print("\n")

                predict.sort_values(by=['Precision(test)'], ascending=False, inplace=True)

                sns.barplot(x='Precision(test)', y='Model', data=predict, palette='Reds_d', ax=axes[1])
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_title('Test Precisions for the Models - White Wine')
                axes[1].set_xlabel('Precision (Test)', size=16)
                axes[1].set_ylabel('Model')
                axes[1].set_xlim(0, 1.0)
                axes[1].set_xticks(np.arange(0, 1.1, 0.1))

                plt.show()
            elif option == 6:

                from sklearn import datasets, tree
                from sklearn.metrics import confusion_matrix
                from sklearn.model_selection import train_test_split

                print("White Wine")
                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print(clf.score(X_test, y_test) * 100, "%")
                print(confusion_matrix(y_test, y_pred))

                print("Red Wine")
                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print(clf.score(X_test, y_test) * 100, "%")
                print(confusion_matrix(y_test, y_pred))

            elif option == 7:
                print("\n")
                print("Red Wine")
                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                rf = RandomForestClassifier(max_depth=10, random_state=0)
                clf = rf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print(clf.score(X_test, y_test) * 100, "%")
                print(confusion_matrix(y_test, y_pred))

                print("\n")
                print("White Wine")
                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                rf = RandomForestClassifier(max_depth=10, random_state=0)
                clf = rf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print(clf.score(X_test, y_test) * 100, "%")
                print(confusion_matrix(y_test, y_pred))

            elif option == 8:
                from sklearn import svm, datasets
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import plot_confusion_matrix

                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

                np.set_printoptions(precision=2)

                # Plot non-normalized confusion matrix
                titles_options = [("Confusion matrix, without normalization", None),
                                  ("Normalized confusion matrix", 'true')]
                for title, normalize in titles_options:
                    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                                 cmap=plt.cm.Blues,
                                                 normalize=normalize)
                    disp.ax_.set_title(title)

                    print(title)
                    print(disp.confusion_matrix)

                plt.show()

            elif option == 9:
                from sklearn import tree

                print("\n ")
                print("\n White Wine")
                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()

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
                clf = rf.fit(df_white)
                centroids = clf.cluster_centers_
                accuracykmc = silhouette_score(df_white, clf.labels_) * 100
                print(centroids)
                print(accuracykmc, "%")

                print("Perceptron Classification")

                scaler = MinMaxScaler()
                scaler.fit(df_white.iloc[:, 0:11])
                scaled_features = scaler.transform(df_white.iloc[:, 0:11])
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
                y = df_white.iloc[:, 11:12].values.ravel()

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
                          ('K-Means Clustering ', accuracykmc),
                          ('Perceptron Classification', accuracypc)]

                predict = pd.DataFrame(data=models, columns=['Model', 'Accuracy'])
                print(predict)

                f, axe = plt.subplots(1, 1, figsize=(18, 6))

                predict.sort_values(by=['Accuracy'], ascending=False, inplace=True)

                sns.barplot(x='Accuracy', y='Model', data=predict, ax=axe)
                # axes[0].set(xlabel='Region', ylabel='Charges')
                axe.set_xlabel('Score in %', size=16)
                axe.set_ylabel('Model')

                plt.show()

            elif option == 10:
                from sklearn import tree

                print("\n ")
                print("\n Red Wine")
                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()

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
                clf = rf.fit(df_red)
                centroids = clf.cluster_centers_
                accuracykmc = silhouette_score(df_red, clf.labels_) * 100
                print(centroids)
                print(accuracykmc, "%")

                print("Perceptron Classification")

                scaler = MinMaxScaler()
                scaler.fit(df_red.iloc[:, 0:11])
                scaled_features = scaler.transform(df_red.iloc[:, 0:11])
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
                y = df_red.iloc[:, 11:12].values.ravel()

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

                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()

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
                print("Perceptron Classification")
                clf = Perceptron(verbose=3)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Predicting Cross Validation Score
                cv_pc = cross_val_score(estimator=clf, X=X_train_scaled, y=y_train.ravel(), cv=5)
                print("CV:", cv_pc)
                print("MEAN:", cv_pc.mean())

                tableScores = {'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
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
                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()

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

                tableScores = {'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
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

                X = df_white.iloc[:, 0:11].values
                y = df_white.iloc[:, 11:12].values.ravel()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                rf = k_means = cluster.KMeans(n_clusters=2)
                clf = rf.fit(df_white)
                centroids = clf.cluster_centers_
                score = silhouette_score(df_white, clf.labels_)
                print(centroids)
                print(score)

            elif option == 14:
                from sklearn import cluster
                from sklearn.metrics import silhouette_score

                X = df_red.iloc[:, 0:11].values
                y = df_red.iloc[:, 11:12].values.ravel()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                rf = k_means = cluster.KMeans(n_clusters=2)
                clf = rf.fit(df_red)
                centroids = clf.cluster_centers_
                score = silhouette_score(df_red, clf.labels_)
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
                scaler.fit(df_red.iloc[:, 0:11])
                scaled_features = scaler.transform(df_red.iloc[:, 0:11])
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
                y = df_red.iloc[:, 11:12].values.ravel()

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
                scaler.fit(df_white.iloc[:, 0:11])
                scaled_features = scaler.transform(df_white.iloc[:, 0:11])
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
                y = df_white.iloc[:, 11:12].values.ravel()

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

                    X = df_red.iloc[:, 0:11].values
                    y = df_red.iloc[:, 11:12].values.ravel()

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

                    X = df_red.iloc[:, 0:11].values
                    y = df_red.iloc[:, 11:12].values.ravel()

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
                    X = df_red.iloc[:, 0:11].values
                    y = df_red.iloc[:, 11:12].values.ravel()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                    clf = tree.DecisionTreeClassifier()
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    print(clf.score(X_test, y_test) * 100, "%")
                    print(confusion_matrix(y_test, y_pred))

                def rf():
                    print("\n")
                    print("Red Wine")
                    X = df_red.iloc[:, 0:11].values
                    y = df_red.iloc[:, 11:12].values.ravel()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                    rf = RandomForestClassifier(max_depth=10, random_state=0)
                    clf = rf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    print(clf.score(X_test, y_test) * 100, "%")
                    print(confusion_matrix(y_test, y_pred))

                def k_means():

                    from sklearn import cluster
                    from sklearn.metrics import silhouette_score

                    X = df_red.iloc[:, 0:11].values
                    y = df_red.iloc[:, 11:12].values.ravel()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                    rf = k_means = cluster.KMeans(n_clusters=2)
                    clf = rf.fit(df_red)
                    centroids = clf.cluster_centers_
                    score = silhouette_score(df_red, clf.labels_)
                    print(centroids)
                    print(score)

                def perceptron():
                    from sklearn.linear_model import Perceptron
                    from sklearn.metrics import confusion_matrix
                    from sklearn.model_selection import train_test_split

                    scaler = MinMaxScaler()
                    scaler.fit(df_red.iloc[:, 0:11])
                    scaled_features = scaler.transform(df_red.iloc[:, 0:11])
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
                    y = df_red.iloc[:, 11:12].values.ravel()

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

                tableScores = {'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
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

                tabelatempos = {'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
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

                    X = df_white.iloc[:, 0:11].values
                    y = df_white.iloc[:, 11:12].values.ravel()

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

                    X = df_white.iloc[:, 0:11].values
                    y = df_white.iloc[:, 11:12].values.ravel()

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
                    X = df_white.iloc[:, 0:11].values
                    y = df_white.iloc[:, 11:12].values.ravel()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

                    clf = tree.DecisionTreeClassifier()
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    print(clf.score(X_test, y_test) * 100, "%")
                    print(confusion_matrix(y_test, y_pred))


                def rf():
                    print("\n")
                    print("White Wine")
                    X = df_white.iloc[:, 0:11].values
                    y = df_white.iloc[:, 11:12].values.ravel()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                    rf = RandomForestClassifier(max_depth=10, random_state=0)
                    clf = rf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    print(clf.score(X_test, y_test) * 100, "%")
                    print(confusion_matrix(y_test, y_pred))

                def k_means():
                    from sklearn import cluster
                    from sklearn.metrics import silhouette_score

                    X = df_white.iloc[:, 0:11].values
                    y = df_white.iloc[:, 11:12].values.ravel()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                    rf = k_means = cluster.KMeans(n_clusters=2)
                    clf = rf.fit(df_red)
                    centroids = clf.cluster_centers_
                    score = silhouette_score(df_red, clf.labels_)
                    print(centroids)
                    print(score)

                def perceptron():
                    from sklearn.linear_model import Perceptron
                    from sklearn.metrics import confusion_matrix
                    from sklearn.model_selection import train_test_split

                    scaler = MinMaxScaler()
                    scaler.fit(df_white.iloc[:, 0:11])
                    scaled_features = scaler.transform(df_white.iloc[:, 0:11])
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
                    y = df_white.iloc[:, 11:12].values.ravel()

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

                tableScores = {'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
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

                tabelatempos = {'Model': ['K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Decision Tree Classification',
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
            submenu3()
            option = int(input("Enter the command you want to execute:"))
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