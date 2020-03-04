### CBI Project

<p align="center"> <img src="./data/figures/streamlit.gif"/> </p>

Using Beautiful Soup to scrape down dollar auction data from the [Central Bank of Iraq](https://www.cbi.iq/) and then visualizing it in Streamlit. Goal is to see if there have been any noticeable changes in auction amounts over the past two years.

See writeup [here](https://medium.com/@mpokornyus/iranian-exploitation-of-iraqs-dollar-auction-3391af5032e0) detailing scraping and visualizations.

#### Contents:

* data:
  * figures - screenshots of sample auction results and plotly visualizations
  * raw - raw scraped data in csv format
  * processed - cleaned data in csv format
* notebooks:
  * cbi_viz - plotting auction amounts over time for the first writeup
  * iforest_cbi - applying the Isolation Forest algorithm to the data and visualizing the results
* python:
  * scraper_range1.py - script to scrape the first range of CBI data
  * scraper_range2.py - script to scrape the second range of CBI data
  * process.py - script to take the raw scraped data and output a cleaned dataframe
  * streamlit_iforest.py - script that ecompasses both the visualization and modeling from the notebooks in a streamlit app

### Setup
* clone the repo
* with cd being the repo, run `pipenv install Pipfile` to set up the environment
* `pipenv shell` will activate the environment
* to launch the streamlit app, run `streamlit run python/streamlit_iforest.py`
