#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
import copy
import seaborn as sns


feature_indexes=[5,6,7,8,9,10]

def MakeLinearRegression(x,y,normalize=False):
    columns = pd.DataFrame(x).columns.shape[0]
    X=np.array(x).reshape([-1,columns])
    
    if normalize:
        X=X-np.mean(X,axis=0)
    
    model = LinearRegression().fit(X,y)
    return model


def TestLinearModel(model,x,normalize=False):
    columns = pd.DataFrame(x).columns.shape[0]
    X=np.array(x).reshape([-1,columns])
    
    if normalize:
        X=X-np.mean(X,axis=0)

    return model.predict(X)


def BackwardElimination(x,y):
    columns = pd.DataFrame(x).columns.shape[0]
    if columns==1:
        return
    
    res={}

    for col in x.columns:
        X=x.drop(columns=col)
        model=MakeLinearRegression(X, y)
        res[col]=np.sum((y-model.predict(X))**2)
    
    key=min(res,key=lambda x:res[x])

    return key,res[key]

def ForwardSelection(xCte,xVar,y):
    XCte=np.array(xCte)
    # xVar=np.array(xVar)
    
    columns = pd.DataFrame(x).columns.shape[0]
    if columns==1:
        return
    
    res={}

    for col in xVar.columns:
        X=np.concatenate((XCte,np.array(xVar.loc[:,col]).reshape((-1,1))),axis=1)
        model=MakeLinearRegression(X, y)
        res[col]=np.sum((y-model.predict(X))**2)
    
    key=min(res,key=lambda x:res[x])

    return key,res[key]

#%% Backward elimination
data = pd.read_csv("data/2020.csv")
data.set_index('Country name',inplace=True)
x=data.iloc[:,feature_indexes]
y=data['Ladder score']

bestModel = MakeLinearRegression(x,y)
loss = np.sum((y-bestModel.predict(x))**2)

labels=['- null']
losses=[loss]

print(x.keys())
X=copy.deepcopy(x)

while len(X.keys())>1:
    k,v = BackwardElimination(X, y)
    labels.append(k)
    losses.append(v)
    X=X.drop(columns=k)
    print(k,v)

labels.append(X.columns[0])
loss=np.sum((y-np.mean(y))**2)
losses.append(loss)

# plt.bar(labels,losses)
plt.figure(figsize=(10,10))
ax=plt.subplot(111)
labels=['all the features','- Generosity','- Corruption','- GDP','- Freedom','- Support','- Health']
plt.scatter(labels,losses)
plt.plot(labels,losses)
plt.xticks(rotation=90,color='r')
# [t.set_color('green') for t in ax.xaxis.get_ticklabels()]
ax.xaxis.get_ticklabels()[0].set_color('green')
plt.title('Backward Elimination result')
plt.ylabel('MSE loss')
plt.xlabel('Elimination')
plt.grid()
plt.savefig('result/BackwardElimination.png')
plt.show()
plt.close()

#%% Forward Selection
data = pd.read_csv("data/2020.csv")
data.set_index('Country name',inplace=True)
x=data.iloc[:,[5,6,7,8,9,10]]
y=data['Ladder score']

labels=['-null']
loss=np.sum((y-np.mean(y))**2)
losses=[loss]

print(x.keys())
xVar=copy.deepcopy(x)
xCte=np.zeros([153,0])

while len(xVar.keys())>1:
    k,v = ForwardSelection(xCte, xVar, y)
    labels.append(k)
    losses.append(v)
    a=np.array(xVar[k]).reshape((-1,1))
    xCte=np.concatenate((xCte,a),axis=1)
    xVar=xVar.drop(columns=k)
    print(k,v)

bestModel = MakeLinearRegression(x,y)
loss = np.sum((y-bestModel.predict(x))**2)

labels.append(xVar.keys()[0])
losses.append(loss)


# plt.bar(labels,losses)
plt.figure(figsize=(10,10))
ax=plt.subplot(111)
labels=['none of the features','+ GDP','+ Freedom','+ Support','+ Health','+ Corruption','+ Generosity']
plt.scatter(labels,losses)
plt.plot(labels,losses)
plt.xticks(rotation=90,color='g')
# [t.set_color('green') for t in ax.xaxis.get_ticklabels()]
ax.xaxis.get_ticklabels()[0].set_color('red')

plt.title('Forward Selection result')
plt.ylabel('MSE loss')
plt.xlabel('Selection')
plt.grid()
plt.savefig('result/ForwardSelection.png')
plt.show()
plt.close()

#%%
data = pd.read_csv("data/2020.csv")
data.set_index('Country name',inplace=True)
x=data.iloc[:,[1,5,6,7,8,9,10,18]]
sns.heatmap(x.corr())
plt.title('Parameters corelation')
plt.savefig('result/corr.png',dpi=50)

#%%

x=data.iloc[:,feature_indexes]
y=data['Ladder score']
bestModel = MakeLinearRegression(x,y)
iran=np.array(data.loc['Iran'][feature_indexes])
# for i in range(len(feature_indexes)):
z=np.ones(len(feature_indexes))
z[5]*=0.85
a=iran*z
# print(iran)
# data.loc['Iran']
print(bestModel.predict(iran.reshape([1,-1])),'---',iran)
print(bestModel.predict(a.reshape([1,-1])),'---',a)
print(data.columns[feature_indexes])


#%%
import numpy as np 
import pandas as pd 
import os
# print(os.listdir("../input"))
import statsmodels.formula.api as stats
from statsmodels.formula.api import ols
import sklearn
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
import plotly
# import plotly.plotly as py 
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

happiness_2015 = pd.read_csv("data/2015.csv")
happiness_2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',
       'Standard Error', 'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']

columns_2015 = ['Region', 'Standard Error']
new_dropped_2015 = happiness_2015.drop(columns_2015, axis=1)

happiness_2016 =  pd.read_csv("data/2016.csv")
columns_2016 = ['Region', 'Lower Confidence Interval','Upper Confidence Interval' ]
dropped_2016 = happiness_2016.drop(columns_2016, axis=1)
dropped_2016.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']

happiness_2017 =  pd.read_csv("data/2017.csv")
columns_2017 = ['Whisker.high','Whisker.low' ]
dropped_2017 = happiness_2017.drop(columns_2017, axis=1)
dropped_2017.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']

frames = [new_dropped_2015, dropped_2016, dropped_2017]
happiness = pd.concat(frames)

data6 = dict(type = 'choropleth', 
           locations = happiness['Country'],
           locationmode = 'country names',
           z = happiness['Happiness_Rank'], 
           text = happiness['Country'],
          colorscale = 'Viridis', reversescale = False)
layout = dict(title = 'Happiness Rank Across the World')#, 
             # geo = dict(showframe = False, 
             #           projection = {'type': 'Mercator'}))
choromap6 = go.Figure(data = [data6], layout=layout)
# choromap6.update_layout(layout)
# plotly.offline.plot(choromap6)

# iplot(choromap6)
# plotly.offline.plot(choromap6)

data2 = dict(type = 'choropleth', 
           locations = happiness['Country'],
           locationmode = 'country names',
           z = happiness['Happiness_Score'], 
           text = happiness['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Happiness Score Across the World')#, 
             # geo = dict(showframe = False, 
             #           projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data2], layout=layout)
choromap3 .update_layout(title_text='Happiness Score Across the World', title_x=0.5)
plotly.offline.plot(choromap3)
# iplot(choromap3)
#%%

data2015=pd.read_csv("archive/2015.csv")
data2016=pd.read_csv("archive/2016.csv")
data2017=pd.read_csv("archive/2017.csv")

for data in [data2015,data2016,data2017]:
    labels=[]
    losses=[]
    x=data.iloc[:,-7:]
    y=data['Happiness Score']

    print('-----------------------')
    print(x.keys(),'\n')

    while len(x.keys())>1:
        k,v = BackwardElimination(x, y)
        labels.append(k)
        losses.append(v)
        x=x.drop(columns=k)
        print(k,v)
    
    plt.bar(labels,losses)
    plt.xticks(rotation=90)
    plt.show()
    plt.close()
        
    
#%%

for col in data2016.columns[-7:] :
    
    plt.Figure(figsize=[10,10])
    for data in [data2015,data2016,data2017]:
        x=data[col]
        y=data['Happiness Score']
        plt.scatter(x,y,alpha=0.7)
        
        model = MakeLinearRegression(x,y)
        x=x.sort_values()
        yPred = TestLinearModel(model, x)
        plt.plot(x,yPred)

    plt.legend(['15','16','17'])
    plt.title(col)
    plt.show()
    plt.close()
    

#%%



x=data2016.iloc[:,-1]
y=data2016['Happiness Score']
model = MakeLinearRegression(x,y)

model.intercept_

#%%
col='Freedom'
x=data2015.iloc[:,-4:]
X= np.array(data2017[col]-data2017[col].mean())
X=np.concatenate((X,np.ones(X.shape))).reshape([2,-1]).T
Y=np.array(data2017['Happiness Score'])

model = LinearRegression().fit(X,Y)

#%%
for col in data2016.columns[-7:] :
    plt.figure(figsize=(5,10))
    plt.suptitle(col)
    
    plt.subplot(311)
    plt.hist(data2015[col],20)
    # plt.xlim([0,1.6])
    
    plt.subplot(312)
    plt.hist(data2016[col],20)
    # plt.xlim([0,1.6])
    
    plt.subplot(313)
    plt.hist(data2017[col],20)
    # plt.xlim([0,1.6])
    
