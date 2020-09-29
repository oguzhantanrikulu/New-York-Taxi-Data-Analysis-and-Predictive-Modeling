#!/usr/bin/env python
# coding: utf-8

# ### Technical test in Data Science

# ### Candidate: Oğuzhan Tanrıkulu

# ### imports: 
# ### For handling with data pandas and numpy, for visualization matplotlib, for dealing with dates datetime and for handling z-score scipy.stats libraries are imported.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import zscore,stats
from sklearn.model_selection import train_test_split


# # Data Loading

# In[2]:


df1 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2009-json_corrigido.json", lines=True)
df2 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2010-json_corrigido.json", lines=True)
df3 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2011-json_corrigido.json", lines=True)
df4 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2012-json_corrigido.json", lines=True)


# In[3]:


df_v = pd.read_csv("../input/nytaxi/data-vendor_lookup-csv.csv")
df_p = pd.read_csv("../input/nytaxi/data-payment_lookup-csv.csv", skiprows = 1)


# # Preparing the data and overview
# ### Combining data in a single dataframe.

# In[4]:


df_all=pd.concat([df1,df2,df3,df4])


# ### Overview of shape, column, and column types in data

# In[5]:


df_all.info()


# ### Displaying the first and last 5 rows of data

# In[6]:


df_all.head(-5)


# ### Overview of Payment lookup data

# In[7]:


df_p


# ### Merging with the main data via 'payment_type' primary key, to use in the future

# In[8]:


df_all = pd.merge(df_all, df_p, on='payment_type')


# ### Overview of Vendor lookup data

# In[9]:


df_v


# ### Preparing vendor lookup data to merge with main data by specifying the column names

# In[10]:


df_v.columns=['vendor_id','vendor_name','vendor_address','vendor_city','vendor_state','vendor_zip','vendor_country','vendor_contact','vendor_current']


# In[11]:


df_v


# ### Merging with the main data via 'vendor_id' as primary key

# In[12]:


df_all = pd.merge(df_all, df_v, on='vendor_id')


# In[13]:


df_all.head(-5)


# ### type of "dropoff_datetime" is converted datetime64 to make it workable

# In[14]:


df_all["dropoff_datetime"]=df_all["dropoff_datetime"].astype("datetime64")


# ### Overview the stats of all data

# In[15]:


df_all.describe()


# # Answer of Question 1

# ### 1. What is the average distance traveled by trips with a maximum of 2 passengers

# ### Locating the mean of trip distances where the passenger count is equal or less then 2

# In[16]:


avd = df_all[['trip_distance']].where(df_all.passenger_count<=2).mean().iloc[0]


# ### Result

# In[17]:


print("\n\nThe average distance traveled by trips with a maximum of 2 passengers is:\n{}".format(avd))


# # Answer of Question 2

# ### 2. Which are the 3 biggest vendors based on the total amount of money raised

# ###  Viewing unique/distinct items to understand the vendors data

# In[18]:


df_all.vendor_id.unique()


# ### So if we get sum of total_amount's of the rides by grouping them by the vendor_id we can reach the results sorted

# In[19]:


df_all['total_amount'].groupby(df_all.vendor_id).sum().sort_values()[::-1]


# ### Because of the vendor_id and vendor_name are unique in the same way we can see the 3 biggest sum amount of vendors with their names

# In[20]:


df_all['total_amount'].groupby(df_all.vendor_name).sum().sort_values()[::-1][:3]


# ### If we want to call the id's name information from an external table

# In[21]:


vdf = df_all['total_amount'].groupby(df_all.vendor_id).sum().sort_values().reset_index()


# In[22]:


pd.merge(vdf[['vendor_id','total_amount']], df_v[['vendor_id','vendor_name']], on='vendor_id')[::-1][:3]


# In[23]:


dfv_xy=pd.merge(vdf[['vendor_id','total_amount']], df_v[['vendor_id','vendor_name']], on='vendor_id')[::-1][:3].iloc[:,[2,1]]


# In[24]:


dfv_xy


# ### Also, the three largest companies can be seen on a bar chart.

# In[25]:


dfv_xy.plot.bar(x='vendor_name', y='total_amount')


# # Answer of Question 3

# ### 3. Make a histogram of the monthly distribution over 4 years of rides paid with cash

# ### Viewing the distinct values of payment type and payment lookup maches

# In[26]:


df_all.payment_type.unique()


# In[27]:


df_all.payment_lookup.unique()


# ### So we can use 'Cash' in payment_lookup as key to find cashes only

# ### Dropoff time is considered ride time

# In[28]:


df_all.dropoff_datetime


# In[29]:


df_all["dropoff_datetime"]


# ### Rides per month for all payment methods

# In[30]:


df_all.dropoff_datetime.groupby(df_all["dropoff_datetime"].dt.month).count().plot(kind="bar",title ='Rides per month for all payment methods' )
plt.xlabel("Month Numbers")
plt.ylabel("Number Of Rides")


# In[31]:


df_all.dropoff_datetime.where(df_all.payment_lookup=='Cash').groupby(df_all["dropoff_datetime"].dt.month).count().plot(kind="bar",title ='Rides per month for Cash payment methods')
plt.xlabel("Month Numbers")
plt.ylabel("Number Of Rides")


# In[32]:


df_all.dropoff_datetime.where(df_all.payment_lookup=='Cash').groupby(by=[(df_all["dropoff_datetime"].dt.year),(df_all["dropoff_datetime"].dt.month)]).count().plot(figsize=(15, 4),kind="bar",title ='Rides per month for each year, for Cash payment methods')
plt.xlabel("Months")
plt.ylabel("Number Of Rides")


# # Answer of Question 4

# ### 4. Make a time series chart computing the number of tips each day for the last 3 months of 2012

# ### Checking whether the trips that are not tipped are null or 0
# 

# In[33]:


df_all.tip_amount.unique()


# In[34]:


df_all.tip_amount.isnull().sum()


# ### There is no null values but 0 values for the trips that are not tipped

# ### Let's see the number of trips that are not tipped

# In[35]:


df_all.tip_amount.where(df_all.tip_amount==0).count()


# ### And the number of trips that are tipped

# In[36]:


df_all.tip_amount.where(df_all.tip_amount!=0).count()


# ### Number of trips that are tipped in the year 2012

# In[37]:


df_all.tip_amount.where((df_all.tip_amount!=0) & (df_all["dropoff_datetime"].dt.year==2012)).count()


# ### Number of trips that are tipped in the last given three months of the year 2012 

# In[38]:


df_all.tip_amount.where((df_all.tip_amount!=0) & (df_all["dropoff_datetime"].dt.year==2012)& (df_all["dropoff_datetime"].dt.month>=(8))).count()


# ### Let's calculate the last three months of 2012 in given data, instead of manually writing it

# In[39]:


months_of_the_year=(df_all["dropoff_datetime"].dt.month).where((df_all["dropoff_datetime"].dt.year==2012)&((df_all["dropoff_datetime"].dt.month)!=np.nan)).unique()


# ### And that gives us the third-to-last month of 2012 so that we can deal with the months that is and after it.

# In[40]:


last3thMonth=sorted(months_of_the_year)[::-1][:3][-1]


# ### So we can use the third-to-last month for detecting number of tips in 2012 last three months

# In[41]:


DaysOfLastThreeMonthsOnCondition=df_all.tip_amount.groupby(df_all["dropoff_datetime"].dt.date.where((df_all.tip_amount!=0) & (df_all["dropoff_datetime"].dt.year==2012)& (df_all["dropoff_datetime"].dt.month>=(last3thMonth)))).count()


# ### Number of tips each day for the last 3 months of 2012

# In[42]:


DaysOfLastThreeMonthsOnCondition


# ### Final graphs: The number of tips each day for the last 3 months of 2012

# In[43]:


DaysOfLastThreeMonthsOnCondition.plot(figsize=(16, 4), title="The number of tips each day for the last 3 months of 2012")
plt.xticks()
plt.xlabel("Dates")
plt.ylabel("Tip Amount")

plt.subplots_adjust(bottom=0.15)
plt.show()


# ### To see each day separately as tics, the series of days in the conditions

# In[44]:


days=sorted(df_all["dropoff_datetime"].dt.date.loc[(df_all["dropoff_datetime"].dt.year==2012) & (df_all["dropoff_datetime"].dt.month>=8)].unique())


# In[45]:


DaysOfLastThreeMonthsOnCondition.plot(figsize=(16, 4), title= "The number of tips each day for the last 3 months of 2012")
plt.xticks(days, rotation='vertical')
plt.xlabel("Dates")
plt.ylabel("Tip Amount")
plt.show()


# In[ ]:





# # Bonus items

# # ● What is the average trip time on Saturdays and Sundays;

# ### the average trip time on Saturdays

# In[46]:


df_all[['trip_distance']].where((df_all["dropoff_datetime"].dt.weekday==5)).mean().iloc[0]


# ### the average trip time on the average trip time on Saturdays

# In[47]:


df_all[['trip_distance']].where((df_all["dropoff_datetime"].dt.weekday==6)).mean().iloc[0]


# ### the average trip time on Saturdays and Sundays

# In[48]:


df_all[['trip_distance']].where((df_all["dropoff_datetime"].dt.weekday==5)|(df_all["dropoff_datetime"].dt.weekday==6)).mean().iloc[0]


# # ● Analyse the data to find and prove seasonality

# ### Defining the seasons of the days of the year.

# In[49]:


summer = range(172, 264)
fall = range(264, 355)
spring = range(80, 172)


def season(x):
    if x in summer:
       return 'Summer'

    if x in fall:
       return 'Fall'

    if x in spring:
       return 'Spring'

    else :
       return 'Winter'


# ### Determining the seasons of each ride / row and assigning these values as a new column.

# In[50]:


bins = [0, 91, 183, 275, 366]
labels=['Winter', 'Spring', 'Summer', 'Fall']
doy = df_all["dropoff_datetime"].dt.dayofyear
df_all['SEASONN'] = pd.cut(doy + 11 - 366*(doy > 355), bins=bins, labels=labels)


# In[51]:


df_all['SEASONN']


# ### Total amount of rides grouped by seasons

# ### This graph clearly shows that there are more rides in the spring and summer seasons.

# In[52]:


plt.xticks([i * 1 for i in range(0, 4)])
df_all.dropoff_datetime.groupby(by=[(df_all['SEASONN'])]).count().plot(kind="bar",  color=['gray', 'green', 'green', 'gray'], title="Total amount of rides grouped by seasons")
plt.xlabel("Seasons")
plt.ylabel("Rides")


# ### Let's look at each year separately

# ### In this line graph, we can see that the hills are again in spring and summer.

# In[53]:


plt.xticks([i * 1 for i in range(0, 16)])
df_all.dropoff_datetime.groupby(by=[(df_all["dropoff_datetime"].dt.year),(df_all['SEASONN'])]).count().plot(figsize=(15, 4),title="Total amount of rides grouped by seasons of each year")
plt.xticks(rotation=30)
plt.xlabel("Seasons by year")
plt.ylabel("Rides")


# ### When we look at the colored bars on a yearly basis, we can see that the hills are spring and summer.

# In[54]:


plt.xticks([i * 1 for i in range(0, 16)])
df_all.dropoff_datetime.groupby(by=[(df_all["dropoff_datetime"].dt.year),(df_all['SEASONN'])]).count().plot(figsize=(15, 4), kind="bar",  color=['gray', 'green', 'green', 'gray'],title="Total amount of rides grouped by seasons of each year")
plt.xticks(rotation=30)
plt.xlabel("Seasons by year")
plt.ylabel("Rides")


# # ● Make a latitude and longitude map view of pickups and dropoffs in the year 2010

# ### Since 2010 is requested, we can work with the data set that contains only this which is df2.

# In[55]:


df2.columns


# ### Calculation of frame border values.

# In[56]:


BBox = ((df2.pickup_longitude.min(), df2.pickup_longitude.max(), df2.pickup_latitude.min(), df2.pickup_latitude.max()))


# In[57]:


BBox


# ### Displaying the locations in these borders.

# In[58]:


fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df2.pickup_longitude, df2.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)

ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])


# ### As can be seen, there are outliers in high deviation in the data.

# ### Let's clear these outliers.

# ### Getting location values to a new dataframe.

# In[59]:


locv = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']


# In[60]:


df_loc=df2[locv]


# ### Removing some values based on Z scores

# In[61]:


z_scores = stats.zscore(df_loc)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3.5).all(axis=1)
new_df = df_loc[filtered_entries]


# ### Recalculating the borders

# In[62]:


BBox2 = ((new_df.pickup_longitude.min(), new_df.pickup_longitude.max(), new_df.pickup_latitude.min(), new_df.pickup_latitude.max()))


# In[63]:


fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(new_df.pickup_longitude, new_df.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)

ax.set_xlim(BBox2[0],BBox2[1])
ax.set_ylim(BBox2[2],BBox2[3])


# ### There are still outliers. So we need to clean more.

# In[64]:


z_scores = stats.zscore(new_df)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3.5).all(axis=1)
new_df2 = new_df[filtered_entries]


# ### Recalculating the borders

# In[65]:


BBox3 = ((new_df2.pickup_longitude.min(), new_df2.pickup_longitude.max(), new_df2.pickup_latitude.min(), new_df2.pickup_latitude.max()))


# In[66]:


BBox3


# In[67]:


fig, ax = plt.subplots(figsize = (11,11))
ax.scatter(new_df2.pickup_longitude, new_df2.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)

ax.set_xlim(BBox3[0],BBox3[1])
ax.set_ylim(BBox3[2],BBox3[3])


# ### Finally we got a more meaningful distribution with no that high bias outliers.

# ### Using the boundary values, a real map can be placed in the substrate.

# In[68]:


nymap = plt.imread("../input/nytaxi/MapNY.jpg")


# In[69]:


fig, ax = plt.subplots(figsize = (17,17))
ax.scatter(new_df2.pickup_longitude, new_df2.pickup_latitude, zorder=1, alpha= 1.0, c='b', s=10, label="pickup")
ax.scatter(new_df2.dropoff_longitude, new_df2.dropoff_latitude, zorder=1, alpha= 0.99, c='r', s=5, label="dropoff")
ax.set_title('Pickup & Dropoff locations in 2010')
ax.set_xlim(BBox3[0],BBox3[1])
ax.set_ylim(BBox3[2],BBox3[3])

plt.legend(loc='upper left',fontsize='large')


ax.imshow(nymap, zorder=0, extent = BBox3, aspect= 'equal')


# ### Drop off and pick up points can be viewed on two separate maps

# In[70]:


fig, ax = plt.subplots(ncols=2, figsize = (19,19))

ax[0].scatter(new_df2.pickup_longitude, new_df2.pickup_latitude, zorder=1, alpha= 1.0, c='b', s=10, label="pickup")
ax[1].scatter(new_df2.dropoff_longitude, new_df2.dropoff_latitude, zorder=1, alpha= 0.99, c='r', s=5, label="dropoff")

ax[0].set_title('Pickup locations in 2010')
ax[1].set_title('Dropoff locations in 2010')

ax[0].set_xlim(BBox3[0],BBox3[1])
ax[0].set_ylim(BBox3[2],BBox3[3])

ax[0].legend(loc='upper left',fontsize='large')
ax[1].legend(loc='upper left',fontsize='large')

ax[0].imshow(nymap, zorder=0, extent = BBox3, aspect= 'equal')

ax[1].set_xlim(BBox3[0],BBox3[1])
ax[1].set_ylim(BBox3[2],BBox3[3])


ax[1].imshow(nymap, zorder=0, extent = BBox3, aspect= 'equal')


# # ● Find what the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations

# ### Getting location values to a new dataframe.

# In[71]:


loca = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','fare_amount','tolls_amount']


# In[72]:


df_loc_all=df_all[loca]


# ### Clearing outliers.

# ### Removing some values based on Z scores

# In[73]:


z_scores = stats.zscore(df_loc_all)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3.5).all(axis=1)
new_dfa = df_loc_all[filtered_entries]

z_scores = stats.zscore(new_dfa)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3.5).all(axis=1)
new_dfa2 = new_dfa[filtered_entries]


# In[74]:


fig, ax = plt.subplots(figsize = (11,11))
ax.scatter(new_dfa2.pickup_longitude, new_dfa2.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)
ax.scatter(new_dfa2.dropoff_longitude, new_dfa2.dropoff_latitude, zorder=1, alpha= 0.2, c='b', s=10)


# ### Outliers look quite cleared

# ### Looking at cleared data

# In[75]:


new_dfa2


# ### Determination for features

# ### New feature for the fare amount (inclusive of tolls)

# In[76]:


new_dfa2['fare_and_tolls_amount']=new_dfa2.fare_amount+new_dfa2.tolls_amount


# ### New feature for the distance between pick and drop longitudes

# In[77]:


new_dfa2['longtitude_distance']=abs(new_dfa2.dropoff_longitude-new_dfa2.pickup_longitude)


# ### New feature for the distance between pick and drop latitudes

# In[78]:


new_dfa2['latitude_distance']=abs(new_dfa2.dropoff_latitude-new_dfa2.pickup_latitude)


# ### This hypotenuse value, which gives the air distance between pickup and dropoff locations, will help the model to make it more accurate.

# In[79]:


import math
new_dfa2['hypotenuse']=np.sqrt((new_dfa2['longtitude_distance']**2)+(new_dfa2['latitude_distance']**2))


# ### Determination of dependent and independent variables. 

# In[80]:


X = new_dfa2[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','longtitude_distance','latitude_distance','hypotenuse']].values
y = new_dfa2['fare_and_tolls_amount'].values


# ### The division of data into two to train and test the model.

# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# ### importing sklearn for machine learning

# In[82]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 


# ### Creating a linear regression model with train data of dependent and independent variables

# In[83]:


from sklearn.linear_model import LinearRegression

r = LinearRegression()
r.fit(X_train, y_train)


# ### Accuracy of the model based on r^2 score

# In[84]:


print("Test set R^2 score is: {:.2f}".format(r.score(X_test, y_test)))


# ### Accuracy of the model based on cross validation score

# In[85]:


from sklearn.model_selection import cross_val_score

cross_val_score(r, X_test, y_test, cv=3).mean()


# ### 78% is a good enough score

# ### Creating the estimation function.

# In[86]:


def predictor(pickup_longitudeX, pickup_latitudeX, dropoff_longitudeX, dropoff_latitudeX):
    lodi=abs(pickup_longitudeX-dropoff_longitudeX)
    ladi=abs(pickup_latitudeX-dropoff_latitudeX)
    hypo=math.sqrt((lodi**2)+(ladi**2))
    
    ar=np.array([[pickup_longitudeX, pickup_latitudeX, dropoff_longitudeX, dropoff_latitudeX, lodi, ladi, hypo]])
    return print("Estimated fare is: ${:.2f}".format(r.predict(ar)[0]))


# #### For being realistic:
# #### longtitutes should be between -74.18 and -73.75,
# #### Latitutes should be between 40.50 and 41.02

# ### This is the predictor.

# In[87]:


#predictor(pickup_longitudeX, pickup_latitudeX, dropoff_longitudeX, dropoff_latitudeX):


# In[88]:


predictor(-74, 41,-74, 41)


# In[89]:


predictor(-74, 41,-74, 41.2)


# In[90]:


predictor(-73.948288,40.774511,-73.997466,40.718039)


# ### This is the predictor.

# # ● Create assumptions, validate against a data and prove with storyelling and graphs

# ### Assumption: There is less rides at night than during the day.

# ### Determining the day and night according to the hours of the day.

# In[91]:


during_day = df_all.dropoff_datetime.where((df_all["dropoff_datetime"].dt.hour>18)|(df_all["dropoff_datetime"].dt.hour<6)).dropna()


# In[92]:


during_night = df_all.dropoff_datetime.where((df_all["dropoff_datetime"].dt.hour<=18)|(df_all["dropoff_datetime"].dt.hour>=6)).dropna()


# In[93]:


during_day.groupby(df_all["dropoff_datetime"].dt.dayofyear).count().plot(kind="line",title ='Rides in during day and during night by day of the year', label="During nights")
during_night.groupby(df_all["dropoff_datetime"].dt.dayofyear).count().plot(kind="line", label="During days")
plt.xlabel("Day of the year")
plt.ylabel("Number of rides")
plt.legend(loc='lower center',fontsize='large')


# ### The graph clearly shows that the number of rides at night (from 18:00 to 06:00) is much less than during the day.
