#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
sns.set_style('darkgrid')
import plotly.graph_objects as go
import plotly.express as px
import chart_studio
import chart_studio.plotly as py

from sklearn import preprocessing
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest

st.title("Central Bank of Iraq Dollar Auction Data")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# st.image(load_image("imgs/iris_setosa.jpg"), width=400)
st.subheader("Sample of Auction Results")
st.text("This auction happened on 18 September 2018. ")
@st.cache
def load_image(img):
    im = Image.open(os.path.join(img))
    return im
st.image(load_image("./data/figures/sample_auction.png"),width=600)


def load_data():
    data = pd.read_csv('./data/processed/processed.csv')
    return data
data = load_data()

if st.checkbox('view_data'):
    st.subheader("Sample of the Data")
    st.write(data[0:55])

st.subheader('Total for Covering Foreign Accounts Since September 2017')
@st.cache
def plot_amounts_over_time(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,y=data.total_for_foreign,
                            mode='lines',
                            name='Total for Foreign'))
    fig.add_trace(go.Scatter(x=data.index,y=data.total_for_foreign.rolling(15, win_type='triang').mean(),
                            mode='lines',
                            name='15-session Rolling Average'))

    # Add lines for each bit of sanctions news
    fig.update_layout(
        shapes=[
            # JCPOA withdrawal announcement
            go.layout.Shape(
                type='line',
                x0=164,
                y0=-30000000,
                x1=164,
                y1=300000000,
                line=dict(
                    color="gray",
                    width=3,
                ),
                layer='below',
                opacity=.3
            ),
            # line for 06 Aug 2018 first wave
            go.layout.Shape(
                type='line',
                x0=227,
                y0=-30000000,
                x1=227,
                y1=300000000,
                line=dict(
                    color='gray',
                    width=3,
                ),
                layer='below',
                opacity=.3
            ),
            # Line for full implementation snapback
            go.layout.Shape(
                type='line',
                x0=291,
                y0=-30000000,
                x1=291,
                y1=300000000,
                line=dict(
                    color='gray',
                    width=3
                ),
                layer='below',
                opacity=.3
            )
        ]
    )
    annotations=[
            dict(x=227,
                y=200000000,
                text='6 Aug 2018 Snapback',
                xanchor='center',
                font=dict(family='Arial',
                         size=14,
                         color='gray')),
            dict(x=164,
                y=30000000,
                text='8 May 2018 Announcement',
                xanchor='center',
                font=dict(family='Arial',
                         size=14,
                         color='gray')),
            dict(x=291,
                y=20000000,
                text='4 Nov 2018 Snapback',
                font=dict(family='Arial',
                         size=14,
                         color='gray'),
                align='right',
                )
        ]
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(
        ticktext=["1 Jan 2018","1 Jan 2019","1 Jan 2020"],
        tickvals=[76,334,592]
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        yaxis=dict(
            showgrid=False,
            showline=False
    ))
    fig.update_yaxes(title_text='Amount Auctioned')
    return fig
fig = plot_amounts_over_time(data)
st.plotly_chart(fig)


# Scatter plot to show distribution of the two amounts
st.subheader('Scatter Plot of Total for Foreign and Total Cash')

plt.figure(figsize=(12,10))
plt.scatter(x=data['total_for_foreign'],
            y=data['total_cash'],
            c='royalblue',marker='H'
           )
plt.xlabel('Total for Foreign',labelpad=10,fontsize=13)
plt.ylabel('Total Cash',labelpad=10,fontsize=13)
plt.ticklabel_format(useOffset=False, style='plain')

st.pyplot()



def normalize_and_model(data):
    # Normalize the data
    X = data[['total_for_foreign','total_cash']].dropna()
    X_norm = preprocessing.normalize(X)
    # Fit the model
    clf = IsolationForest(max_samples=100,random_state=42)
    clf.fit(X)


    df = X.copy(deep=True)
    df['anomaly_scores'] = clf.decision_function(X)
    # Add anomaly scores to original dataframe
    data.loc[data.index.isin(df.index),'anomaly_scores'] = df['anomaly_scores']
    return data

data_iforest = normalize_and_model(data)

# Visualize distribution of anomaly scores
st.subheader("Distribution of Anomaly Scores")
plt.figure(figsize=(15,5))
plt.hist(
    x=data_iforest['anomaly_scores'].dropna(),
    bins=60,
    color='royalblue'
)
plt.xlabel("Score",labelpad=10,fontsize=13)
plt.ylabel("Number of Datapoints",labelpad=10,fontsize=13)

st.pyplot()

percentile = st.slider('select_percentile',0,100,10)

@st.cache
def apply_pctile_label(data,percentile):

    data['most_anomalous'] = np.where(
        data.anomaly_scores <= data.anomaly_scores.quantile(percentile/100),
        1,
        0
    )
    return data

labeled_data = apply_pctile_label(data=data_iforest,percentile=percentile)
st.subheader("Scatter Plot with Labels Overlayed")
st.text("User the slider above to adjust the percentile of abnormality.")


markers = ["H",'X']
sizes = [30, 60]
colors= ['royalblue','red']
plt.figure(figsize=(14,7))

for i in range(0,2):
    plt.scatter(
        labeled_data[labeled_data.most_anomalous==i]['total_for_foreign'],
        labeled_data[labeled_data.most_anomalous==i]['total_cash'],
        s=sizes[i],
        marker=markers[i],
        c=colors[i]
    )
plt.xlabel('Total for Foreign',labelpad=10,fontsize=13)
plt.ylabel('Total Cash',labelpad=10,fontsize=13)
plt.legend(
    ('Bottom {}% Least Anomalous'.format(100-percentile),'Top {}% Most Anomalous'.format(percentile)),
    loc='upper right',
    fontsize=15
)
plt.ticklabel_format(useOffset=False, style='plain')
st.pyplot()


# In[10]:


st.write(data.sort_values(by='anomaly_scores').reset_index().iloc[0:20,1:-1])
