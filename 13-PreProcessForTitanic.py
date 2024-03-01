import numpy as np
import pandas as pd

train_data = pd.read_csv("dataset/titanic/Titanic_Train.csv")
test_data = pd.read_csv("dataset/titanic/Titanic_Test.csv")

# 只截取这些特征 因为目前只能处理数字
features=['Pclass','Age','SibSp','Parch','Fare']

output = pd.DataFrame({'PassengerId': train_data.PassengerId,
                       'Pclass':train_data.Pclass,
                       'SibSp':train_data.SibSp,
                        'Parch':train_data.Parch,
                       'Fare':train_data.Fare,},dtype=np.float32)
output.to_csv('train.csv', index=False)