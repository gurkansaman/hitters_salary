
#GÖREV:1
#######################################
# Hitters
#######################################
# Değişkenler
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör


# Importing Library:

import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,validation_curve
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.eda import *
from helpers.data_prep import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from helpers.data_prep import *
from helpers.eda import *

# Setting:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#Importing File:
df = pd.read_csv("datasets/hitters.csv")
df.head()

################
# EDA
################
# (322, 20)
# NewLeague;Division;League kadar "object" türündedir:
# Ön kontrol: Salary 59 adet NA değeri vardır:

check_df(df)

# Numerik, kategorik değişkenlere ayıralım:
cat_cols, num_cols, cat_but_car =grab_col_names(df)

###Salary Değişkeni Hitter missing value var bunları knn ile dolduralım:

df_knn = df.select_dtypes(include=["float64","int64"])
imputer = KNNImputer(n_neighbors=15)
df_knn = imputer.fit_transform(df_knn)
df_knn = pd.DataFrame(df_knn,columns=num_cols)
df["Salary"] = df_knn["Salary"]


# Num_col histogram olarak gözlemleyelim:
for col in num_cols:
    plt.hist(df[col], align='mid',color = "skyblue")
    plt.title(col)
    plt.show()

# Aşağıkdaki değerler gözlemlere dayalıdır. Yani Outlier bu değerlerden kırpılabilir:
# Salary < 1500
# CRBI < 1500
# CRuns <1500
# CHits < 3000

df.shape
df.info()

# Gözlemsel olarak outlier df içinden çıkaralım RMSE minimize etmede önemli olacaktır
df = df[(df['Salary'] < 1500) & (df['CHits']<3000)
        & (df["CRBI"]< 1500) & (df["CRuns"]<1500)]
########################################
#New Feature
########################################

    df['AtBat*RBI'] = df['AtBat'] * df['RBI']
    df['Walks*Years'] = df['Walks'] * df['Years']
    df['AtBat*RBI'] = df['AtBat'] * df['RBI']
    df['Walks*Years'] = df['Walks'] * df['Years']
    df['AtBat/Hits'] = df['AtBat'] / df['Hits']
    df['AtBat/Runs'] = df['AtBat'] / df['Runs']
    df['Hits/Runs'] = df['Hits'] / df['Runs']
    df['HmRun/RBI'] = df['HmRun'] / df['RBI']
    df['Runs/RBI'] = df['Runs'] / df['RBI']
    df['Years/CAtBat'] = df['Years'] / df['CAtBat']
    df['Years/CHits'] = df['Years'] / df['CHits']
    df['Years/CHmRun'] = df['Years'] / df['CHmRun']
    df['Years/CRuns'] = df['Years'] / df['CRuns']
    df['Years/CRBI'] = df['Years'] / df['CRBI']
    df['CAtBat/CHits'] = df['CAtBat'] / df['CHits']
    df['CAtBat/CRuns'] = df['CAtBat'] / df['CRuns']
    df['CAtBat/CRBI'] = df['CAtBat'] / df['CRBI']
    df['CAtBat/CWalks'] = df['CAtBat'] / df['CWalks']
    df['CHits/CRuns'] = df['CHits'] / df['CRuns']
    df['CHits/CRBI'] = df['CHits'] / df['CRBI']
    df['CHits/CWalks'] = df['CHits'] / df['CWalks']
    df['CHmRun/CRuns'] = df['CHmRun'] / df['CRuns']
    df['CHmRun/CRBI'] = df['CHmRun'] / df['CRBI']
    df['CHmRun/CWalks'] = df['CHmRun'] / df['CWalks']
    df['CRuns/CRBI'] = df['CRuns'] / df['CRBI']
    df['CRuns/CWalks'] = df['CRuns'] / df['CWalks']
    df['CHmRun/CRBI'] = df['CHmRun'] / df['CRBI']

df.replace([np.inf, -np.inf], 0, inplace=True)

# Columns kategorize edelim:
cat_cols, num_cols, cat_but_car =grab_col_names(df)

#cat_cols: 3
#num_cols: 17
#cat_but_car: 0
#num_but_cat: 0

# Outlier liste hazırlanması:

outlier_list= []
for col in num_cols:
    if check_outlier(df, col) == True:
        outlier_list.append(col)
        print(col, check_outlier(df, col))

outlier_list

# Replace thresholds for outlier values:
for col in num_cols:
    replace_with_thresholds(df,col)

#Missing Value
missing_values_table(df)
df.dropna(subset=["HmRun/RBI","CHmRun/CWalks","CHmRun/CRBI"], axis=0, inplace=True)

# 0.9 ve üstü korelasyonu olan feature silinmesi:
# Regrosyon konusunda multicor konusu önemlidir o nedenle eşbaskınlıklar önüne geçilir.
drop_list = high_correlated_cols(df,corr_th=0.80)
drop_list

# Silme fonksiyonu
df.drop(drop_list, axis=1, inplace=True)

#Kontrol edilmesi
df.shape
df.info()
df.head()

# cat_cols dummy veya binary formata göre labelması:
### Label encoding ###
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
# ['League', 'Division', 'NewLeague']
for col in binary_cols:
    df = label_encoder(df, col)

# Kontrol edilmesi
df.shape
df.info()
df.head()

#Robust Scale:
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove("Salary") #Target değişkeni çıkardık

for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

# Kontrol edilmesi
df.shape
df.info()
df.head()

# EDA sonuçlanması:
y = df["Salary"]
X = df.drop(["Salary"], axis=1)

# .7 ile .3 oranında train ve test olarak da ayıralım:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)



######################################################
# Base Models
######################################################

# Önce base model default değerler ile ölçelim:

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          #("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ('ExtraTrees', ExtraTreesRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# RMSE Hata Score:
print("######### RMSE Hata Score ###############")
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y,
                                            cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# R2 Score:
print("######### R2 Score ###############")
for name, regressor in models:
    r2 = np.mean(cross_val_score(regressor, X, y,
                                            cv=10, scoring="r2"))
    print(f"R2: {round(r2, 4)} ({name}) ")


# ortlama ile rmse karşılaştırılmalı hatta std ile de bakılmalı
y.mean()
y.std()



######################################################
# Automated Hyperparameter Optimization
######################################################

# Parametrelerin tanımlanması:

    cart_params = {'max_depth': range(1, 20),
                "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [5, 8, 15, None],
                "max_features": [5, 7, "auto"],
                "min_samples_split": [8, 15, 20],
                 "n_estimators": [200, 500]}

    svr_params = param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
                          'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],
                          'gamma' : ('auto','scale')}

#xgboost_params = {"learning_rate": [0.01, 0.05, 0.1, 0.15],
    #                "max_depth": [3, 5, 8],
    #                "n_estimators": [100, 200, 300],
    #                "colsample_bytree": [0.3, 0.5, 0.8]}

    lightgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
                    "n_estimators": [500, 1000],
                    "colsample_bytree": [0.1, 0.3, 0.7, 1]}


    catboost_params = {"iterations": [500,1000],
                     "learning_rate": [0.01, 0.1],
                     "depth": [3, 6]}

    extraTrees_params = {
        'n_estimators': [500, 1000],
        'max_depth': [2, 16, 50],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'warm_start': [True, False],
     }

    #HistGradient_params = {"learning_rate": [0.01, 0.05],
    #                    "max_iter": [20, 100],
    #                    "max_depth": [None, 25],
    #                    "l2_regularization": [0.0, 1.5],
    #                    }

    regressors = [("CART", DecisionTreeRegressor(), cart_params),
                ("RF", RandomForestRegressor(), rf_params),
     #           ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
                ('LightGBM', LGBMRegressor(), lightgbm_params),
                  ('Catboost',CatBoostRegressor(verbose=False),catboost_params),
                ('ExtraTrees', ExtraTreesRegressor(), extraTrees_params),
     #           ('HistGradient', HistGradientBoostingRegressor(), HistGradient_params)
                ]
#############################
# For döngüsünde test edelim:

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    # Parametrelerin tune edilmesi
    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    # Best after modelin valide edilip, rmse hesaplanması
    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

########## CART ##########
#RMSE: 268.6636 (CART)
#RMSE (After): 237.1565 (CART)
#CART best params: {'max_depth': 5, 'min_samples_split': 2}

########## RF ##########
#RMSE: 197.918 (RF)
#RMSE (After): 198.5132 (RF)
#RF best params: {'max_depth': None, 'max_features': 'auto', 'min_samples_split': 8, 'n_estimators': 500}

########## LightGBM ##########
#RMSE: 195.9382 (LightGBM)
#RMSE (After): 193.4625 (LightGBM)
#LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 500}

########## Catboost ##########
#RMSE: 193.2944 (Catboost)
#RMSE (After): 192.925 (Catboost)
#Catboost best params: {'depth': 6, 'iterations': 1000, 'learning_rate': 0.01}

########## ExtraTrees ##########
#RMSE: 190.2604 (ExtraTrees)
#RMSE (After): 188.7849 (ExtraTrees)
#ExtraTrees best params: {'bootstrap': False, 'max_depth': 50, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500, 'warm_start': True}

######################################################
# # Stacking & Ensemble Learning
######################################################

# Voting model nesnesinin tanımlanması:
voting_reg = VotingRegressor(estimators=[('ExtraTrees', best_models["ExtraTrees"]),
                                         ('LightGBM', best_models["LightGBM"]),
                                         ('Catboost', best_models['Catboost'])])

# Model nesnesinin train edilmesi:
voting_reg.fit(X, y)

# Model başarasının test edilmesi:
np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

#RMSE: 189.38589318529074


# GÖREV:2
###########################################################################################
# LightGBM modelinin karmaşıklığının tüm parametreler açısından learning Curve incelenmesi:
############################################################################################



def val_curve_params(model, X, y, param_name, param_range, scoring="neg_mean_squared_error", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)


    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Parameter: {param_name}")
    plt.ylabel(f"Scoring: RMSE")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Parametreleri tanımlayalım:

lgbm_params = [["learning_rate", [0.001,0.01,0.1]],
               ["n_estimators", [100, 300, 500, 1000]],
               ["colsample_bytree", [0.5, 0.7, 1]]]


# Model nesnesini tanımlayalım:
lgbm_model = LGBMRegressor(random_state=17)

# Her parametre için kotrol edelim:
for i in range(len(lgbm_params)):
    val_curve_params(lgbm_model, X, y, lgbm_params[i][0], lgbm_params[i][1])