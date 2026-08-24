from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', None)
data = pd.read_csv('vgsales.csv')
data.info()

print(data.isnull().sum())
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
data.info()

numbers_only = data.select_dtypes(['number']).columns
characters_only = data.select_dtypes(['string' ,'object']).columns

preprocessing = ColumnTransformer(
    transformers=[(
        'Numbers_only',
        Pipeline([
            ('simple', SimpleImputer(strategy='median')),
            ('encoder',StandardScaler())
        ]),
     numbers_only
    ),
        (
            'Characters_only',
            Pipeline([
                ('simple', SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(handle_unknown='ignore'))
            ]),
            characters_only
        )
    ])

x = data
y = data['Global_Sales']

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42

)

Linear_Regression  = Pipeline([
    ('preprocessing', preprocessing),
    ('classifier', LinearRegression())
])
Decision_Tree = Pipeline([
    ('preprocessing', preprocessing),
    ('classifier', DecisionTreeRegressor())
])

set1 = Decision_Tree.fit(x_train,y_train)
nothing1 = set1.predict(x_test)
nothing1_accuracy = r2_score(y_test, nothing1)

set = Linear_Regression.fit(x_train,y_train)
nothing = set.predict(x_test)
nothing_accuracy = r2_score(y_test,nothing)
print(f'Prediction for Linear regression is {nothing} and Accuracy is {nothing_accuracy}')
print(f'Prediction for Decision Tree is {nothing1} and Accuracy is {nothing1_accuracy}')