import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
pd.options.plotting.backend = "plotly"

st.title('Steam games analysis')
st.write('I chose a dataset to analyze from kaggle.com, it\'s calles \'Steam Games Dataset\' The file shows scraped data about the video games from steam (2023, Sep). There are several features for each game which was initially gathered for the purposes of creating recommendation system. The dataset is uncleaned, and it consists of 71700 rows and 16 columns. It contains:')

st.markdown('''1. Title: Showing the title of the game
2. Original Price: Price before discount
3. Discounted Price: Price after discount
4. Release Date: The release date of the game
5. Link: link to the game page on steam
6. Game Description: Short description on the page of the game on steam
7. Recent Reviews Summary: Summary of the reviews according to the recent feedback
8. All Reviews Summary: Summary of the reviews according to the all time feedback
9. Recent Reviews Number: Count of the review
10. All Reviews Number: Count of the review
11. Developer
12. Publisher
13. Supported Languages
14. Popular Tags
15. Game Features
16. Minimum Requirements''')
st.write('Firsty, import all the libraries to work with')
st.code('''import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"''')

st.write('Creating a dataframe from a dataset')
steamGamesDF = pd.read_csv("merged_data.csv")
st.dataframe(steamGamesDF)

st.write('Now let\'s look at the column data types')
st.code('steamGamesDF.info()')
buffer = io.StringIO()
steamGamesDF.info(buf=buffer)
s = buffer.getvalue()
st.text(s)
buffer.close()
st.code('''steamGamesDF.isna().sum()''')
st.text(steamGamesDF.isna().sum())
st.title('Data cleanup')
st.write('Replace all empty fields with "Not stated"')
st.code('''steamGamesDF.fillna("Not stated", inplace=True)
steamGamesDF.head(4)''')
steamGamesDF.fillna("Not stated", inplace=True)
st.dataframe(steamGamesDF)
st.write('First, let\'s check the dataset for duplicates')
steamGamesDF = steamGamesDF.drop_duplicates()

st.code('''steamGamesDF = steamGamesDF.drop_duplicates()
len(steamGamesDF)''')
st.write(len(steamGamesDF))

st.markdown('''As you can see, the number of rows has not changed, which means there are no duplicates in the dataset

Now we can noticed that columns "Original Price","Discounted Price","Release Date" have dtype object, but it is wrong
"Original Price" must be float
"Discounted Price" must be float
Let's change this''')

def convertPriceToFloat(m):
    m = str(m)
    if m == 'Free':
        return 0
    else:
        m = m.replace(',', '')
        return float(m[1::])

st.code('''def convertPriceToFloat(m):
    m = str(m)
    if m == 'Free':
        return 0
    else:
        m = m.replace(',', '')
        return float(m[1::])''')


steamGamesDF['Original Price'] = [convertPriceToFloat(i) for i in steamGamesDF['Original Price']]
steamGamesDF['Discounted Price'] = [convertPriceToFloat(i) for i in steamGamesDF['Discounted Price']]

st.code('''steamGamesDF['Original Price'] = [convertPriceToFloat(i) for i in steamGamesDF['Original Price']]
steamGamesDF['Discounted Price'] = [convertPriceToFloat(i) for i in steamGamesDF['Discounted Price']]
steamGamesDF.head(4)''')

st.dataframe(steamGamesDF.head(4))

st.write('Now I want to delete games with a price exceeding $ 1000, because they are not suitable for analysis')
st.code('''toDel = steamGamesDF[steamGamesDF['Original Price'] > 1000].index
steamGamesDF.loc[toDel]''')
toDel = steamGamesDF[steamGamesDF['Original Price'] > 1000].index
st.dataframe(steamGamesDF.loc[toDel])
st.write('Next, we remove them from the dataset')
st.code('''steamGamesDF.drop(toDel, inplace = True)
toDel = steamGamesDF[steamGamesDF['Discounted Price'] > 1000].index
steamGamesDF.drop(toDel, inplace = True)''')
steamGamesDF.drop(toDel, inplace = True)
toDel = steamGamesDF[steamGamesDF['Discounted Price'] > 1000].index
steamGamesDF.drop(toDel, inplace = True)
st.write('''Then I want to replace column 'Recent Reviews Summary' with lowercase values with numeric values. To do this, let's see what data is stored in the column''')
st.code('''print(steamGamesDF['Recent Reviews Summary'].unique())''')
st.write(steamGamesDF['Recent Reviews Summary'].unique())
st.markdown('''I want to change the data in the h column to numeric because in this way
0 - Empty data or the number of reviews, becouse the number of users does not allow us to determine the totality of reviews due to the fact that there are too few of them
1 - Overwhelmingly Negative 
2 - Very Negative 
3 - Negative 
4 - Mostly Negative 
5 - Mixed 
6 - Mostly Positive 
7 - Positive 
8 - Very Positive 
9 - Overwhelmingly''')
st.code('''def convertReviewsToInt(x):
    x = str(x)
    if x == 'Overwhelmingly Negative':
        return 1
    elif x == 'Very Negative':
        return 2
    elif x == 'Negative':
        return 3
    elif x == 'Mostly Negative':
        return 4
    elif x == 'Mixed':
        return 5
    elif x == 'Mostly Positive':
        return 6
    elif x == 'Positive':
        return 7
    elif x == 'Very Positive':
        return 8
    elif x == 'Overwhelmingly Positive':
        return 9
    else:
        return 0''')
def convertReviewsToInt(x):
    x = str(x)
    if x == 'Overwhelmingly Negative':
        return 1
    elif x == 'Very Negative':
        return 2
    elif x == 'Negative':
        return 3
    elif x == 'Mostly Negative':
        return 4
    elif x == 'Mixed':
        return 5
    elif x == 'Mostly Positive':
        return 6
    elif x == 'Positive':
        return 7
    elif x == 'Very Positive':
        return 8
    elif x == 'Overwhelmingly Positive':
        return 9
    else:
        return 0


st.write('I am deleting data in which \'Recent Reviews Summary\' = 0')
st.code('''steamGamesDF['Recent Reviews Summary'] = [convertReviewsToInt(i) for i in steamGamesDF['Recent Reviews Summary']]
toDel = steamGamesDF[steamGamesDF['Recent Reviews Summary'] == 0].index
steamGamesDF.drop(toDel, inplace = True)''')
steamGamesDF['Recent Reviews Summary'] = [convertReviewsToInt(i) for i in steamGamesDF['Recent Reviews Summary']]
toDel = steamGamesDF[steamGamesDF['Recent Reviews Summary'] == 0].index
steamGamesDF.drop(toDel, inplace = True)

st.write('Next I want to determine the year of release for each game and remove all games that have not yet been released.')
st.code('''def convertToYearOrDel(x):
    s = x.split()
    for i in s:
        f = str(i)
        if len(f) == 4 and f.isdigit():
            f = int(f)
            if f <= 2023:
                return int(f)
    return \'toDel\'''')

def convertToYearOrDel(x):
    x = str(x)
    x = x.split()
    for i in x:
        f = str(i)
        if len(f) == 4 and f.isdigit():
            f = int(f)
            if f <= 2023:
                return int(f)
    return 'toDel'

st.code('''steamGamesDF['Release Date'] = [convertToYearOrDel(i) for i in steamGamesDF['Release Date']]
steamGamesDF.head(4)''')
steamGamesDF['Release Date'] = [convertToYearOrDel(i) for i in steamGamesDF['Release Date']]
st.dataframe(steamGamesDF.head(4))
st.write('Now I will delete all the inappropriate data')
st.code('''toDel = steamGamesDF[steamGamesDF['Release Date'] == 'toDel'].index
steamGamesDF.drop(toDel, inplace = True)
steamGamesDF['Release Date'] = steamGamesDF['Release Date'].astype(int)''')

toDel = steamGamesDF[steamGamesDF['Release Date'] == 'toDel'].index
steamGamesDF.drop(toDel, inplace = True)
steamGamesDF['Release Date'] = steamGamesDF['Release Date'].astype(int)

st.write('Next, I need to get the percentage of positive reviews from the column \'Recent Reviews Number\'')
st.code('''def convertToPercent(x):
    s = x.split('%')
    if len(s) == 2:
        f = str(s[0])
        f = f[-3::].split(' ')
        x = int(f[len(f) - 1])
        if x<=100:
            return x
    return \'toDel\'''')

def convertToPercent(x):
    x = x.split('%')
    if len(x) == 2:
        f = str(x[0])
        f = f[-3::].split(' ')
        x = int(f[len(f) - 1])
        if x<=100:
            return x
    return 'toDel'
st.code('''steamGamesDF['Recent Reviews Number'] = [convertToPercent(i) for i in steamGamesDF['Recent Reviews Number']]
toDel = steamGamesDF[steamGamesDF['Recent Reviews Number'] == 'toDel'].index
steamGamesDF.drop(toDel, inplace = True)''')
steamGamesDF['Recent Reviews Number'] = [convertToPercent(i) for i in steamGamesDF['Recent Reviews Number']]
toDel = steamGamesDF[steamGamesDF['Recent Reviews Number'] == 'toDel'].index
steamGamesDF.drop(toDel, inplace = True)
st.code('''steamGamesDF.info()''')
buffer = io.StringIO()
steamGamesDF.info(buf=buffer)
s = buffer.getvalue()
st.text(s)
st.code('''steamGamesDF.isna().sum() ''')
st.text(steamGamesDF.isna().sum())
st.write('We can see that the amount of data has almost halved')
st.title('Overview')

st.code('''meanPrice = steamGamesDF['Original Price'].mean()
medianPrice = steamGamesDF['Original Price'].median()
standardDeviationPrice  = steamGamesDF['Original Price'].std()

print(f'Mean: {meanPrice}')
print(f'Median: {medianPrice}')
print(f'Standard: {standardDeviationPrice}')''')
meanPrice = steamGamesDF['Original Price'].mean()
medianPrice = steamGamesDF['Original Price'].median()
standardDeviationPrice  = steamGamesDF['Original Price'].std()
st.write(f'Mean: {meanPrice}')
st.write(f'Median: {medianPrice}')
st.write(f'Standard: {standardDeviationPrice}')
st.code('''meanRS = steamGamesDF['Recent Reviews Summary'].mean()
medianRS = steamGamesDF['Recent Reviews Summary'].median()
standardDeviationRS  = steamGamesDF['Recent Reviews Summary'].std()

print(f'Mean: {meanRS}')
print(f'Median: {medianRS}')
print(f'Standard: {standardDeviationRS}')''')
meanRS = steamGamesDF['Recent Reviews Summary'].mean()
medianRS = steamGamesDF['Recent Reviews Summary'].median()
standardDeviationRS  = steamGamesDF['Recent Reviews Summary'].std()

st.write(f'Mean: {meanRS}')
st.write(f'Median: {medianRS}')
st.write(f'Standard: {standardDeviationRS}')

st.code('''meanRN = steamGamesDF['Recent Reviews Number'].mean()
medianRN = steamGamesDF['Recent Reviews Number'].median()
standardDeviationRN  = steamGamesDF['Recent Reviews Number'].std()

print(f'Mean: {meanRN}')
print(f'Median: {medianRN}')
print(f'Standard: {standardDeviationRN}')''')
meanRN = steamGamesDF['Recent Reviews Number'].mean()
medianRN = steamGamesDF['Recent Reviews Number'].median()
standardDeviationRN  = steamGamesDF['Recent Reviews Number'].std()

st.write(f'Mean: {meanRN}')
st.write(f'Median: {medianRN}')
st.write(f'Standard: {standardDeviationRN}')

st.write('Now let\'s plot the dependence of the average price on the year of release of the game')
st.code('''a = steamGamesDF.groupby("Release Date")['Original Price'].mean()''')
a = steamGamesDF.groupby("Release Date")['Original Price'].mean()
st.code('''fig2 = px.bar(a, y='Original Price')
# fig2 = a.plot.area()
fig2.update_layout(title="Dependence of the average original price on the year of release",showlegend = False)
fig2.show()''')

fig2 = px.bar(a, y='Original Price')
# fig2 = a.plot.area()
fig2.update_layout(title="Dependence of the average original price on the year of release",showlegend = False)
st.plotly_chart(fig2,theme=None)
st.write('I decided that in addition to this, I can additionally build an average discounted price depending on the year of release of the game.Which will allow me to compare the original price with the discounted price.')
st.code('''b = steamGamesDF.groupby("Release Date")['Discounted Price'].mean()
c =  pd.DataFrame({'Release Date':list(a.keys())+list(b.keys()),'Price type':list(['Original']*len(a))+list(['Discounte']*len(b)),'Price':[a[i] for i in a.keys()] + [b[i] for i in b.keys()]})
fig = px.line(c,x='Release Date', y='Price', color='Price type', height=400)
fig.update_layout(title="Dependence of the average price on the year of release")
fig.show()''')
b = steamGamesDF.groupby("Release Date")['Discounted Price'].mean()
c =  pd.DataFrame({'Release Date':list(a.keys())+list(b.keys()),'Price type':list(['Original']*len(a))+list(['Discounte']*len(b)),'Price':[a[i] for i in a.keys()] + [b[i] for i in b.keys()]})
fig = px.line(c,x='Release Date', y='Price', color='Price type', height=400)
fig.update_layout(title="Dependence of the average price on the year of release")
st.plotly_chart(fig,theme=None)
st.markdown('''From this graph, you can see that there are no discounts for games before 2004. In addition, from 2016-2018, the prices for games were the lowest. It can also be noted that since 2020 the original price has increased significantly compared to the discount price.

Let's make a graph of the dependence of the average оriginal price on the total user rating''')
st.code('''a = steamGamesDF.groupby("Recent Reviews Summary")['Original Price'].mean()
fig2 = px.bar(a,y='Original Price',color_discrete_sequence=[['Tomato','Coral','DarkOrange','Gold','Yellow','LightGreen','Chartreuse','LimeGreen','Green']])
fig2.update_layout(title="Dependence of the average original price on the reviews summary",showlegend = False)
fig2.show()''')
a = steamGamesDF.groupby("Recent Reviews Summary")['Original Price'].mean()
fig2 = px.bar(a,y='Original Price',color_discrete_sequence=[['Tomato','Coral','DarkOrange','Gold','Yellow','LightGreen','Chartreuse','LimeGreen','Green']])
fig2.update_layout(title="Dependence of the average original price on the reviews summary",showlegend = False)
st.plotly_chart(fig2,theme=None)
st.title('My hypothesis')
st.markdown('''More expensive games have a higher percentage of positive reviews

To test my hypothesis, I will add an additional column "Price range" to the dataframe. It will have the values: 'Free', '0-5', '5-10', '10-20', '20-60', '60+\'''')
st.code('''def createPriceRange(x):
    if x == 0.00:
        return 'Free'
    elif x > 0 and x <= 5:
        return '0-5'
    elif x > 5 and x <= 10:
        return '5-10'
    elif x > 10 and x <= 20:
        return '10-20'
    elif x > 20 and x <= 60:
        return '20-60'
    else:
        return '60+\'''')

def createPriceRange(x):
    if x == 0.00:
        return 'Free'
    elif x > 0 and x <= 5:
        return '0-5'
    elif x > 5 and x <= 10:
        return '5-10'
    elif x > 10 and x <= 20:
        return '10-20'
    elif x > 20 and x <= 60:
        return '20-60'
    else:
        return '60+'

st.code('''steamGamesDF['Price range'] = [createPriceRange(i) for i in steamGamesDF['Original Price']]
steamGamesDF.head(4)''')
steamGamesDF['Price range'] = [createPriceRange(i) for i in steamGamesDF['Original Price']]
st.dataframe(steamGamesDF.head(4))

st.code('''a = steamGamesDF.groupby("Price range")['Recent Reviews Number'].mean()
fig2 = px.bar(a,y='Recent Reviews Number',color_discrete_sequence=[['Blue','Blue','Blue','Blue','Red','Blue']],width=600,height=450)
fig2.update_layout(title="Dependence of the average recent reviews number on the price range")
fig2.show()''')
a = steamGamesDF.groupby("Price range")['Recent Reviews Number'].mean()
fig2 = px.bar(a,y='Recent Reviews Number',color_discrete_sequence=[['Blue','Blue','Blue','Blue','Red','Blue']],width=600,height=450)
fig2.update_layout(title="Dependence of the average recent reviews number on the price range")
st.plotly_chart(fig2,theme=None)
st.write('As we can see, too expensive games have the lowest percentage of positive votes, which refutes the hypothesis')
st.title('Languages')
st.write('I want to find out what are the most popular languages that games have been translated into')
st.code('''def createLanguageList(x):
    x = str(x)
    s = ','
    x = x.replace('\"', s)
    x = x.replace('\'', s)
    x = x.replace('[', s)
    x = x.replace(']', s)
    t = x.split(s)
    r = []
    for i in t:
        if len(i) > 1:
            r.append(i)
    if len(r) != 0:
        return list(r)
    else:
        return 'Game has not supported languages\'''')

def createLanguageList(x):
    x = str(x)
    s = ','
    x = x.replace('\"', s)
    x = x.replace('\'', s)
    x = x.replace('[', s)
    x = x.replace(']', s)
    t = x.split(s)
    r = []
    for i in t:
        if len(i) > 1:
            r.append(i)
    if len(r) != 0:
        return list(r)
    else:
        return 'Game has not supported languages'

st.code('''steamGamesDF['Supported Languages'] = [createLanguageList(i) for i in steamGamesDF['Supported Languages']]''')
steamGamesDF['Supported Languages'] = [createLanguageList(i) for i in steamGamesDF['Supported Languages']]
st.code('''d = dict()
for i in steamGamesDF['Supported Languages']:
    if i =='Game has not supported languages':
        continue
    i = list(i)
    for j in i:
        if j in d.keys():
            d[j] += 1
        else:
            d[j] = 1''')
d = dict()
for i in steamGamesDF['Supported Languages']:
    if i =='Game has not supported languages':
        continue
    i = list(i)
    for j in i:
        if j in d.keys():
            d[j] += 1
        else:
            d[j] = 1
st.code('''SupLangDf = pd.DataFrame({'Supported Languages': d.keys(),'Count': d.values()})
SupLangDf''')
SupLangDf = pd.DataFrame({'Supported Languages': d.keys(),'Count': d.values()})
st.dataframe(SupLangDf)
st.write('Now let\'s combine non-popular languages into "Other languages"')
st.code('''df = SupLangDf.copy(deep=True)
df.loc[df['Count'] < 6000, 'Supported Languages'] = 'Other language'
pie = px.pie(df,values='Count', names='Supported Languages')
pie.update_traces(textposition='inside', textinfo='percent+label',hole=.1,pull=[0.1 for i in range(0,109) if i==1])
pie.update_layout(title="Supported languages")
pie.show()''')
df = SupLangDf.copy(deep=True)
df.loc[df['Count'] < 6000, 'Supported Languages'] = 'Other language'
pie = px.pie(df,values='Count', names='Supported Languages')
pie.update_traces(textposition='inside', textinfo='percent+label',hole=.1,pull=[0.1 for i in range(0,109) if i==1])
pie.update_layout(title="Supported languages")
st.plotly_chart(pie,theme=None)
st.title('Tags')
st.write('I want to use column \'Popular Tags\', and I create a list of tags for every game')
st.code('''def createTagList(x):
    x = str(x)
    s = ','
    x = x.replace('t \'e', 't e')
    x = x.replace('\"', s)
    x = x.replace('\'', s)
    x = x.replace('[', s)
    x = x.replace(']', s)
    t = x.split(s)
    r = []
    for i in t:
        if len(i) > 1:
            r.append(i)
    if len(r) != 0:
        return list(r)
    else:
        return 'Game has not tags\'''')
def createTagList(x):
    x = str(x)
    s = ','
    x = x.replace('t \'e', 't e')
    x = x.replace('\"', s)
    x = x.replace('\'', s)
    x = x.replace('[', s)
    x = x.replace(']', s)
    t = x.split(s)
    r = []
    for i in t:
        if len(i) > 1:
            r.append(i)
    if len(r) != 0:
        return list(r)
    else:
        return 'Game has not tags'

st.code('''steamGamesDF['Popular Tags'] = [createTagList(i) for i in steamGamesDF['Popular Tags']]''')
steamGamesDF['Popular Tags'] = [createTagList(i) for i in steamGamesDF['Popular Tags']]
st.markdown('''I am creating a new dataframe with columns: 'Name', 'Count', 'Original Price', 'Recent Reviews Number\'''')
st.code('''d = dict()
for i in steamGamesDF['Popular Tags']:
    if i =='Game has not tags':
        continue
    i = list(i)
    for j in range(0,min(len(i),2)):
        if i[j] in d.keys():
            d[i[j]] += 1
        else:
            d[i[j]] = 1''')
d = dict()
for i in steamGamesDF['Popular Tags']:
    if i =='Game has not tags':
        continue
    i = list(i)
    for j in range(0,min(len(i),2)):
        if i[j] in d.keys():
            d[i[j]] += 1
        else:
            d[i[j]] = 1
st.code('''g = steamGamesDF[['Popular Tags','Original Price','Recent Reviews Number']].values.tolist()
d2 = dict()
d3 = dict()
for i in d.keys():
    d2[i]=0.0 
    d3[i]=0.0
for i in g:
    for j in range(0,min(len(i[0]),2)):
        x = i[0][j]
        if(x in d):
            d2[x] += i[1]
            d3[x] += i[2]''')
g = steamGamesDF[['Popular Tags','Original Price','Recent Reviews Number']].values.tolist()
d2 = dict()
d3 = dict()
for i in d.keys():
    d2[i]=0.0
    d3[i]=0.0
for i in g:
    for j in range(0,min(len(i[0]),2)):
        x = i[0][j]
        if(x in d):
            d2[x] += i[1]
            d3[x] += i[2]
st.code('''for i in d.keys():
    d2[i]= d2[i]/d[i]
    d3[i]= d3[i]/d[i]''')
for i in d.keys():
    d2[i]= d2[i]/d[i]
    d3[i]= d3[i]/d[i]

st.code('''TagsDf = pd.DataFrame({'Name': d.keys(),'Count': d.values(),'Original Price': d2.values(),'Recent Reviews Number': d3.values()})
TagsDf''')
TagsDf = pd.DataFrame({'Name': d.keys(),'Count': d.values(),'Original Price': d2.values(),'Recent Reviews Number': d3.values()})
st.dataframe(TagsDf)
st.write('I will remove all non-popular tags and replace them with "Other tags"')
st.code('''df = TagsDf.copy(deep=True)
df.loc[df['Count'] < 1500, 'Name'] = 'Other tags' ''')
df = TagsDf.copy(deep=True)
df.loc[df['Count'] < 1500, 'Name'] = 'Other tags'
st.write('Now let\'s output a pie chart of tags')

st.code('''pie = px.pie(df,values='Count', names='Name')
pie.update_traces(textposition='inside', textinfo='percent+label')
pie.update_layout(title="Tags")
pie.show()''')
pie = px.pie(df,values='Count', names='Name')
pie.update_traces(textposition='inside', textinfo='percent+label')
pie.update_layout(title="Tags")
st.plotly_chart(pie,theme=None)
st.title('More detailed overview')
st.write('Now we will output the average price and the average percentage of positive reviews for most popular tags')
st.code('''df = TagsDf.copy(deep=True)
df.loc[df['Count'] < 400, 'Name'] = 'Other tags' 
toDel = df[df['Name'] == 'Other tags'].index
df.drop(toDel, inplace = True)''')
df = TagsDf.copy(deep=True)
df.loc[df['Count'] < 400, 'Name'] = 'Other tags'
toDel = df[df['Name'] == 'Other tags'].index
df.drop(toDel, inplace = True)
st.code('''fig = px.bar(df, x='Name', y='Original Price', color='Recent Reviews Number', height=400, color_continuous_scale=px.colors.sequential.Plotly3)
fig.update_layout(title="Tags price and reviews number")
fig.show()''')
fig = px.bar(df, x='Name', y='Original Price', color='Recent Reviews Number', height=400, color_continuous_scale=px.colors.sequential.Plotly3)
fig.update_layout(title="Tags price and reviews number")
st.plotly_chart(fig,theme=None)
st.markdown('''You can see that the Anime tag is the most expensive. The Simulations tag has the lowest percentage of positive reviews, while Novels and Arcades have the highest.

The dependence of the total score and the number of positive reviews for each year of release of games''')
st.code('''fig = px.density_contour(
    steamGamesDF,
    y='Recent Reviews Number',
    x='Release Date',
    z='Recent Reviews Summary',
    histfunc="max",
    nbinsx=30,
    nbinsy=30,
)
fig.update_traces(contours_coloring="fill", contours_showlabels = True)
fig.update_layout(title="Dependence of the total score and the number of positive reviews for each year of release of games")
fig.show()''')
fig = px.density_contour(
    steamGamesDF,
    y='Recent Reviews Number',
    x='Release Date',
    z='Recent Reviews Summary',
    histfunc="max",
    nbinsx=30,
    nbinsy=30,
)
fig.update_traces(contours_coloring="fill", contours_showlabels = True)
fig.update_layout(title="Dependence of the total score and the number of positive reviews for each year of release of games")
st.plotly_chart(fig,theme=None)
st.markdown('''From this graph, you can see that the higher the percentage of positive reviews, the higher the total score.And I made such an addiction.

- 3: 2-21
- 4: 17-42
- 5: 37-72
- 6: 67-79
- 7: 77-81
- 8: 80-94
- 9: 92-100''')




















