# README for NYC Taxi Data Analysis

---

## Overview

This project provides a comprehensive analysis of the New York City taxi data over the years 2009-2012. It aims to handle various data analysis processes and build data pipelines for the New York Taxi Rides data. The main insights include the average distance traveled, top-performing vendors, monthly ride distributions, and more.

## Main notebook file is [FinalCode_v2.0.ipynb](https://github.com/oguzhantanrikulu/NYtaxi/blob/master/FinalCode_v2.0.ipynb)

Python source code file is  [FinalCode_v2.0.py](https://github.com/oguzhantanrikulu/NYtaxi/blob/master/FinalCode_v2.0.py)


## Table of Contents
1. [Data Loading](#data-loading)
2. [Data Preparation](#data-preparation)
3. [Questions and Answers](#questions-and-answers)
4. [Bonus Insights](#bonus-insights)
5. [Assumptions and Validations](#assumptions-and-validations)
6. [Prerequisites](#prerequisites)
7. [Installing Data](#installing-data)

## Data Loading

The primary datasets for this analysis are:
1. NYC Taxi trip data for the years 2009 to 2012.
2. Vendor lookup data.
3. Payment lookup data.

The trip data is loaded from JSON files, while the vendor and payment lookup data are loaded from CSV files.

## Data Preparation

The main steps in the data preparation phase include:

1. Merging the datasets from various years into a single DataFrame.
2. Displaying the shape, columns, and data types of the combined dataset.
3. Merging the main dataset with the payment and vendor lookup data using appropriate keys.
4. Converting specific columns to appropriate data types for easier analysis.

## Questions and Answers

The following questions were posed and subsequently answered:

1. What is the average distance traveled by trips with a maximum of 2 passengers?
2. Which are the three biggest vendors based on the total amount of money raised?
3. Distribution of rides paid with cash on a monthly basis over the years.
4. Time series chart showing the number of tips each day for the last 3 months of 2012.

## Bonus Insights

1. Average trip time on Saturdays and Sundays.
2. Analysis to find and prove seasonality in the data.
3. Latitude and longitude map views of pickups and drop-offs in the year 2010.

## Assumptions and Validations

A primary assumption was validated through analysis:

- There are fewer rides at night than during the day. This assumption was proven correct with the help of a visual graph showing the number of rides during the day vs. during the night.

## Prerequisites

1. Python (minimum version 2.6).
2. Environment: Jupyter Notebook.
3. Libraries: pandas, numpy, matplotlib, seaborn, datetime, scipy.stats, sklearn.
4. Data and image of the New York map.

## Installing Data

The data can be downloaded from the following Kaggle link: [DATA from Kaggle (including map image)](https://www.kaggle.com/oguzhantanrikulu/nytaxi/download)

Please ensure you update the file paths to reflect the location of your data. For instance, the paths in the following code need to be adjusted:

```python
df1 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2009-json_corrigido.json", lines=True)
...
df_p = pd.read_csv("../input/nytaxi/data-payment_lookup-csv.csv", skiprows = 1)
nymap = plt.imread("../input/nytaxi/MapNY.jpg")
```

**Additional Resources:**
- [KAGGLE PAGE OF THE NOTEBOOK](https://www.kaggle.com/oguzhantanrikulu/notebook81c11a26b9)

**Author:** Oğuzhan Tanrıkulu

---
