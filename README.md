# NYtaxi
# Technical test in Data Science
# Candidate: Oğuzhan Tanrıkulu

The goal is to handle various data analysis processes and building data pipelines with New York Taxi Rides data.

Main notebook file is [FinalCode_v2.0.ipynb](https://github.com/oguzhantanrikulu/NYtaxi/blob/master/FinalCode_v2.0.ipynb)

Python source code file is  [FinalCode_v2.0.py](https://github.com/oguzhantanrikulu/NYtaxi/blob/master/FinalCode_v2.0.py)

And the data files are mentioned in the installing data part below.

Also this project is on kaggle page: [KAGGLE PAGE OF THE NOTEBOOK](https://www.kaggle.com/oguzhantanrikulu/notebook81c11a26b9)

Github page of the project [github.com/oguzhantanrikulu/NYtaxi](https://github.com/oguzhantanrikulu/NYtaxi)

This project is also uploaded to Amazon Web Service in SageMaker Studio and also as Notebook instances 

### Prerequisites

This project needs minimum python 2.6 and the environment was Jupyter Notebook.
Related data and image of the new york map are also required.

### Installing data

The data is given bu it also can be downloaded from here: [DATA from kaggle (also map image inclueded)](https://www.kaggle.com/oguzhantanrikulu/nytaxi/download)

File paths need to be changed with the path of that your data. In the notebook file or source code file these paths are needed to change:

```
df1 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2009-json_corrigido.json", lines=True)
df2 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2010-json_corrigido.json", lines=True)
df3 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2011-json_corrigido.json", lines=True)
df4 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2012-json_corrigido.json", lines=True)
df_v = pd.read_csv("../input/nytaxi/data-vendor_lookup-csv.csv")
df_p = pd.read_csv("../input/nytaxi/data-payment_lookup-csv.csv", skiprows = 1)
```

also this path of the map image needs to be changed in the code, with the path of the file that needed to be downloaded:[DOWNLOAD THE MAP IMAGE](doesn't work anymore)

```
nymap = plt.imread("../input/nytaxi/MapNY.jpg")
```
