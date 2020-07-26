#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
sns.set_style('darkgrid')

from sklearn import preprocessing
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest

from bokeh.plotting import figure, show
from bokeh. io import output_notebook
from bokeh.layouts import column, row
from bokeh.models import Span, HoverTool, Label

st.title("Central Bank of Iraq Dollar Auction Data")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# st.image(load_image("imgs/iris_setosa.jpg"), width=400)
st.markdown("""_A short visual analysis of data scraped from the Central Bank
of Iraq's daily dollar auction_""")
st.subheader("Sample of Auction Results")
st.markdown("""This auction happened on September 18, 2018. "Total sales for the
purposes of fortifying foreign accounts" was 190,058,455; "total sales for cash"
was 35,900,000.""")
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
    st.subheader("Sample of the Scraped Data")
    st.write(data[0:55])

st.subheader('Total for Covering Foreign Accounts Since September 2017')
st.markdown("""Plot of the amounts auctioned to cover foreign accounts over the
past 2+ years. Vertical markers indicate significant announcements regarding
the United States exiting the JCPOA agreement.""")
# @st.cache
# def plot_amounts_over_time(data):

plot = figure(
    title='Total for covering foreign accounts since September 2017',
    x_axis_label='Date',
    y_axis_label='Amount',
    plot_width=700,
    plot_height=425
)

plot.line(
    data.index,
    data.total_for_foreign,
    line_width=2,
    line_color='darkblue',
    alpha=0.5,
    legend_label="Total for foreign"
)

plot.line(
    data.index,
    data.grand_total.rolling(7).mean(),
    line_width=2,
    line_color='fuchsia',
    alpha=0.5,
    legend_label="Foreign 7-day rolling avg"
)

# legend
plot.legend.location = "top_left"
plot.legend.click_policy="hide"
plot.legend.background_fill_color = 'white'

# hover
hover = HoverTool(tooltips=[('Amount','$y{int}')],mode='vline')
plot.add_tools(hover)

# line for sanctions announcement
sanct_announce = Span(
    location = 164,
    dimension='height',
    line_color='darkgray',
    line_dash='dashed'
)
plot.add_layout(sanct_announce)

sanct_announce_label = Label(
    x=164,
    y=190000000,
    text='8 May 2018 announcement',
    text_color='gray',
    border_line_color='gray',
    text_font_size='12px',
    text_align='right'
)
plot.add_layout(sanct_announce_label)

# line for first snapback sanctions
first_snapback = Span(
    location = 227,
    dimension='height',
    line_color='darkgray',
    line_dash='dashed'
)
plot.add_layout(first_snapback)

first_snapback_label = Label(
    x=227,
    y=15000000,
    text='6 Aug 2018 snapback sanctions',
    text_color='gray',
    border_line_color='gray',
    text_font_size='12px',
    text_align='left'
)
plot.add_layout(first_snapback_label)

second_snapback = Span(
    location = 291,
    dimension='height',
    line_color='darkgray',
    line_dash='dashed'
)
plot.add_layout(second_snapback)

second_snapback_label = Label(
    x=291,
    y=39000000,
    text='4 Nov 2018 snapback sanctions',
    text_color='gray',
    border_line_color='gray',
    text_font_size='12px',
    text_align='left'
)
plot.add_layout(second_snapback_label)


plot.xaxis.ticker = [76, 334,583]
plot.xaxis.major_label_overrides = {76: '2018-01-01', 334: '2019-01-01',583:'2020-01-01'}

# turn off scientific notation for the y axis
plot.yaxis.formatter.use_scientific = False

# layout = column([plot], sizing_mode='scale_width')
    # return plot

# plot = plot_amounts_over_time(data)
st.bokeh_chart(plot)


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
st.markdown("""After running the Isolation Forest algorithm on the data, below
is the distribution of anomaly scores across the data. The lower the score, the
more anomalous the algorithm has labeled the datapoint.
""")
plt.figure(figsize=(15,7))
plt.hist(
    x=data_iforest['anomaly_scores'].dropna(),
    bins=60,
    color='royalblue'
)
plt.xlabel("Score",labelpad=10,fontsize=15)
plt.xticks(fontsize=14)
plt.ylabel("Number of Datapoints",labelpad=10,fontsize=15)
plt.yticks(fontsize=14)

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
st.markdown("User the slider above to adjust the percentile of abnormality.")


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

st.subheader("Top 20 Most Anomalous Datapoints")
st.markdown("""As was expected, the datapoints that the Isolation Forest algorithm
finds most anomalous are the outlier values in _total_for_foreign_ and _total_cash_.
""")
st.write(data.sort_values(by='anomaly_scores').reset_index().iloc[0:20,1:-1])