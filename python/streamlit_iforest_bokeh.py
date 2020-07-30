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
sns.set_style('whitegrid')

from sklearn import preprocessing
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest

from bokeh.plotting import figure, show
from bokeh. io import output_notebook
from bokeh.layouts import column, row
from bokeh.models import Span, HoverTool, Label, ColumnDataSource

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
    data['rolling_foreign'] = data.total_for_foreign.rolling(7).mean()
    return data
data = load_data()

if st.checkbox('View Data'):
    st.subheader("Sample of the Scraped Data")
    st.write(data[0:55])

st.subheader('Total for Covering Foreign Accounts Since September 2017')
st.markdown("""Plot of the amounts auctioned to cover foreign accounts over the
past 2+ years. Vertical markers indicate significant announcements regarding
the United States exiting the JCPOA.""")

@st.cache
def plot_amounts_over_time(data):

    source1 = ColumnDataSource(data)
    source2 = ColumnDataSource(data)

    plot = figure(
        x_axis_label='Date',
        y_axis_label='Auction Amount',
        plot_width=750,
        plot_height=450
    )
    plot.yaxis.major_label_orientation = np.pi/3

    plot.circle(
        source=source1,
        x="index",
        y="total_for_foreign",
        fill_color="skyblue",
        alpha=.7
    )

    plot.line(
        source=source1,
        x="index",
        y="total_for_foreign",
        line_width=2,
        line_color='dodgerblue',
        alpha=.4,
        legend_label="Total for foreign",
        # hover_color='black',
        # hover_alpha=.8
    )

    plot.line(
        source=source2,
        x="index",
        y="rolling_foreign",
        line_width=2.75,
        line_color='#1E20FF',
        alpha=0.7,
        legend_label="Foreign 7-day rolling avg",
        hover_color='black',
        hover_alpha=.9
    )

    # legend
    plot.legend.location = "top_left"
    plot.legend.click_policy="hide"
    plot.legend.background_fill_color = 'white'
    plot.legend.label_text_font_size = '9pt'

    # hover
    hover = HoverTool(tooltips=[
        ('total for foreign','@total_for_foreign'),
        ('rolling foreign','@rolling_foreign{int}')
    ])
    plot.add_tools(hover)

    # line for sanctions announcement
    sanct_announce = Span(
        location = 164,
        dimension='height',
        line_color='#FF1E90',
        line_width=3,
        line_dash='dashed'
    )
    plot.add_layout(sanct_announce)

    sanct_announce_label = Label(
        x=164,
        y=190000000,
        text='8 May 2018 announcement  ',
        text_color='#FF1E90',
        border_line_color=None,
        text_font_size='12px',
        text_align='right'
    )
    plot.add_layout(sanct_announce_label)

    # line for first snapback sanctions
    first_snapback = Span(
        location = 227,
        dimension='height',
        line_color='#FF1E90',
        line_width=3,
        line_dash='dashed'
    )
    plot.add_layout(first_snapback)

    first_snapback_label = Label(
        x=227,
        y=15000000,
        text='  6 Aug 2018 snapback sanctions',
        text_color='#FF1E90',
        border_line_color=None,
        text_font_size='12px',
        text_align='left'
    )
    plot.add_layout(first_snapback_label)

    second_snapback = Span(
        location = 291,
        dimension='height',
        line_color='#FF1E90',
        line_width=3,
        line_dash='dashed'
    )
    plot.add_layout(second_snapback)

    second_snapback_label = Label(
        x=291,
        y=39000000,
        text='  4 Nov 2018 snapback sanctions',
        text_color='#FF1E90',
        border_line_color=None,
        text_font_size='12px',
        text_align='left'
    )
    plot.add_layout(second_snapback_label)


    plot.xaxis.ticker = [76, 334,583]
    plot.xaxis.major_label_overrides = {76: '2018-01-01', 334: '2019-01-01',583:'2020-01-01'}

    # turn off scientific notation for the y axis
    plot.yaxis.formatter.use_scientific = False
    return plot


plot = plot_amounts_over_time(data)
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

# Visualize distribution of anomaly scores in histogram
st.subheader("Distribution of Anomaly Scores")
st.markdown("""After running the Isolation Forest algorithm on the data, below
is the distribution of anomaly scores across the data. The lower the score, the
more anomalous the algorithm has labeled the datapoint.
""")
def plot_hist(data):

    hist, edges = np.histogram(data,bins=50)

    hist_df = pd.DataFrame({
        "column": hist,
        "left": edges[:-1],
        "right": edges[1:]
    })
    hist_df["interval"] = ["%d to %d" % (left, right) for left,right in zip(hist_df["left"], hist_df["right"])]

    source = ColumnDataSource(hist_df)

    plot = figure(
        plot_height=400,
        plot_width=625,
        x_axis_label='Anomaly score',
        y_axis_label='Count',
    )
    plot.quad(
        top="column",
        bottom=0,
        left="left",
        right="right",
        source=source,
        line_color="black",
        fill_color="dodgerblue",
        alpha=0.9,
        hover_fill_color='red',
        hover_fill_alpha=0.9
    )

    hover = HoverTool(tooltips=[('Count',str("@" + "column"))])
    plot.add_tools(hover)



    return plot

hist = plot_hist(data_iforest['anomaly_scores'].dropna())
st.bokeh_chart(hist)

st.subheader("Scatter Plot with Labels Overlayed")
st.markdown("Use the slider to adjust the percentile of abnormality.")
percentile = st.slider('Select Percentile',0,100,5)

@st.cache
def apply_pctile_label(data,percentile):

    data['most_anomalous'] = np.where(
        data.anomaly_scores <= data.anomaly_scores.quantile(percentile/100),
        1,
        0
    )
    return data

labeled_data = apply_pctile_label(data=data_iforest,percentile=percentile)


markers = ["H",'X']
sizes = [60, 90]
colors= ['dodgerblue','red']
plt.figure(figsize=(14,7))

for i in range(0,2):
    plt.scatter(
        labeled_data[labeled_data.most_anomalous==i]['total_for_foreign'],
        labeled_data[labeled_data.most_anomalous==i]['total_cash'],
        s=sizes[i],
        marker=markers[i],
        c=colors[i],
        alpha=0.8
    )
plt.xlabel('Total for foreign',labelpad=10,fontsize=15)
plt.ylabel('Total cash',labelpad=10,fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(
    ('Bottom {}% least anomalous'.format(100-percentile),'Top {}% most anomalous'.format(percentile)),
    loc='upper right',
    fontsize=16
)
plt.ticklabel_format(useOffset=False, style='plain')
st.pyplot()


# In[10]:

st.subheader("Top 20 Most Anomalous Datapoints")
st.markdown("""As was expected, the datapoints that the Isolation Forest algorithm
finds most anomalous are the outlier values in _total_for_foreign_ and _total_cash_.
""")
st.write(data.sort_values(by='anomaly_scores').reset_index().iloc[0:20,1:-1])
