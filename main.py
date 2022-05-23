# Кітапханаларды импорттадым
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset оқитын функцияларды шақырдым
dftemp = pd.read_csv('dataset.csv')
dftemp = dftemp.set_index('dateRep')



groupped = dftemp.groupby('countriesAndTerritories')

# Dataframe объект жасадым
df = pd.DataFrame()
for name, group in groupped:
    if name == "Kazakhstan":
        df = group

df.info()
# х мәліметтер қатарын алдым
df = df[['day', 'month', 'year', 'cases', 'deaths']]
df = df.iloc[::-1]

# xFrame тәуелсіз айнымалы алдым
xFrame = df.drop('deaths', axis=1)

# y Frame тәуелді айнымалы алдым
yFrame = df['deaths']


# Жаттықтыратын және тесттен өткізетін мәліметтер алдым, тест жасайтын мәлімет көлемі 25%, жаттықтыратын мәлімет 75%
x_train, x_test, y_train, y_test = train_test_split(xFrame.values, yFrame.values, test_size=0.25, shuffle=False)

# Модель құрдым, аты сызықтық регрессия
model = LinearRegression()
model.fit(x_train, y_train) # жаттықтырдым

prediction = model.predict([[21, 5, 2022, 5000]]) # Болжам жасадым
print(prediction)
#print(x_test[0])

#plt.plot(df.index[:-len(prediction)], y_train, label='Training') 
#plt.plot(df.index[-len(prediction):], y_test, label='actual')
#plt.plot(df.index[-len(prediction):], prediction, label='prediction')
#plt.legend(loc=1)
#plt.show()

