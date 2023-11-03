# %%
#Importing all libraries that may be used in order to execute project code.
#The purpose is to build predictive models to determine the popularity of a song on Spotify.
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# %%
#Importing Spotify Dataset. The dataset was opened in Microsoft Excel and the label 'index' was added.
#You will need to have the spotify csv from kaggle linked here: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download
spotifydf = pd.read_csv('')#Include pathname here in between the parentheses
spotifydf

# %%
#The index column is no longer needed as an index has automatically been added, so it will be dropped from the table.
spotifydf = spotifydf.drop(columns=['index'])
spotifydf

# %%
#The following code drops duplicates of artists and tracks since artists may release a track multiple times on different albums, mixtapes, etc. 
spotifydf = spotifydf.drop_duplicates(subset=['artists','track_name'], keep = 'first').reset_index(drop=True)
spotifydf

# %%
#This code gets information on the dataset. Information includes the variable names, count, types, and whether there are null values.
spotifydf.info()

# %%
#This code gets the summary statistics on the numeric variables of the dataset. 
spotifydf.describe()

# %%
#The following code creates boxplots to visualize summary statistics of select numeric variables. Because there are 20 variables,
#the code visualizes three variables: popularity, danceability, and energy.

#Popularity 
popbox = sns.boxplot(spotifydf, x = 'popularity')

# %%
#Danceability
dancebox = sns.boxplot(spotifydf, x = 'danceability')

# %%
#Energy 
energybox = sns.boxplot(spotifydf, x = 'energy')

# %%
#This code provides exploratory visualizations of the categorical variable 'track_genre' since it's not included in the summary statistics.
#Since there are over 100 genres, it looks at the pie chart of the ten most popular genres by sum of popularity.
pie = spotifydf.groupby(['track_genre']).sum().head(10).reset_index().sort_values(by = 'popularity', ascending = False)
piechart = px.pie(pie, names = 'track_genre', values='popularity', title= '10 Most Popular Genres by Sum of Popularity')
piechart.show()

# %%
#This code determines the correlation of the variables in order to visualize and select variables for the predictive model
spotifydf.corr()

# %%
#This a heatmap visualizing the correlation of the variables 
heatmap = px.imshow(spotifydf.corr())
heatmap.show()

# %%
#Looking at the matrix and heatmap, none of the variables seem to be obviously correlated to cause concern for multicollinearity. 
#Therefore primary variables selected for the linear regression predictive were based on personal decisions.
#Selected variables were danceability, energy, and loudness.

# %%
#One of the predictive models is linear regression. This code uses statsmodels to run the linear regression model.
model = smf.ols('popularity ~ danceability + energy +loudness', data = spotifydf)
results = model.fit()
print(results.summary())

# %%
#This is a partial regression plot that displays the relationship between the dependent variable and the given independent varriable
#after removing the effect of the other independent variables.
figlr1 = sm.graphics.plot_partregress_grid(results)

# %%
#This code visualizes the errors (residuals) of the model 
pred_val = results.fittedvalues.copy()
residuals = spotifydf['popularity'] - pred_val
fig = sns.histplot(residuals)
fig

# %%
#This is splitting data for training and testing. 80% goes into training, 20% goes into testing. 
#This will be used to run another linear regression that gives us predictions and for the kNeighbors predictive 
x = spotifydf[['danceability','energy','loudness']].to_numpy()
y = spotifydf['popularity'].to_numpy()
data_train, data_test, label_train, label_test = train_test_split(x,y,test_size=0.2)

# %%
#Another version of the linear Regression using sklearn to get some predictions. 
LinReg = LinearRegression()
LinReg.fit(data_train,label_train)
lrlabel_predict = LinReg.predict(data_test)
lrlabel_predict

# %%
#This is code is checking the R^2 value again to see correlation between the dependent variable and independent variables.
print(LinReg.score(data_test,label_test))

# %%
#This is a visualization of the relationship between predicted and actual values in the sklearn linear regression. It doesn't provide 
#much information
model = LinearRegression().fit(data_train,label_train)
y_pred = model.predict(data_test)
plt.scatter(label_test, y_pred)

# %%
#This code is using kNeighborsRegressors from sklearn as another predictive model. It will predict popularity by
#incorporating associated nearest neighbors. 
#The number of neighbors was manually changed by counting by 5s until reached a point where the model no longer improved significantly.

knn = KNeighborsRegressor(n_neighbors=85)
knn.fit(data_train, label_train)
label_predict = knn.predict(data_test)
label_predict

# %%
#This gives the coefficient of determination of the prediction (like the R^2 value)
print(knn.score(data_test,label_test))

# %%
#This is a visualization of the relationship between predicted and actual values in the sklearn kNeighbors regression. Like the linear
#regression, it doesn't provide much information.
model = knn.fit(data_train, label_train)
y_pred = model.predict(data_test)
plt.scatter(label_test, y_pred)

# %%
#Compared to the linear regression model, the kNeighbors regression model performed better than the linear regression model. 
#The coefficient was 0.05 (rounded) meaning the model detected a higher correlation between the variables. However,in
#both linear regression and kNeighbors regression, the relationship was weak. 
#While the models were able to successfully predict a song's popularity, the selected independent variables may have not been the best 
#to choose. In order to improve the models, other variables would be selected and tested. 


