# NYtaxi
# Technical test in Data Science
# Candidate: Oğuzhan Tanrıkulu

The goal is to handle various data analysis processes and building data pipelines with New York Taxi Rides data.

Main notebook file is [FinalCode_v2.0.ipynb](https://github.com/oguzhantanrikulu/NYtaxi/blob/master/FinalCode_v2.0.ipynb)

Python source code file is  [FinalCode_v2.0.py](https://github.com/oguzhantanrikulu/NYtaxi/blob/master/FinalCode_v2.0.py)

And the data files are mentioned in the installing data part below.

Also this project is on kaggle page: [KAGGLE PAGE OF THE NOTEBOOK](https://www.kaggle.com/oguzhantanrikulu/notebook81c11a26b9)

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

also this path of the map image needs to be changed in the code, with the path of the file that needed to be downloaded:[DOWNLOAD THE MAP IMAGE](https://storage.googleapis.com/kagglesdsdata/datasets/898145/1524458/MapNY.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20200929%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20200929T191050Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=749d2cbd785980fe51dda9ca723907a3fb0ddaa02652ccfb5d0066209bf1ded815e0b87e73962d06ce6c486965145018ce988c063aac277e8fad29d8222d85db5fb3d05cbf45aab1f7a469e3cd5d3b1b1d06ec992cbcbb64364f926773b5e714328c27508584001e3ececf9190c546186c05add70c533de38de1462919d26aa03e63b25f785a34f5d48e9e966c32d80ac55588491331268b223ac9997f4fb166fe8ab8c2861c079cfdaff01479ef34a87f2beb438f9e6ec35657321417f6da7551f07beccfd833f5cb9aae755d9cc37ffc6b593a5f8fda9d23395e52fa26684c5ca7ae608997269f6309d3056f09270eeaca45df36617f3f878f0a680456b292)

```
nymap = plt.imread("../input/nytaxi/MapNY.jpg")
```
