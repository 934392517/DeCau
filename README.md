# DeCau
### 1. Table of Contexts

~~~text
configs					-> training Configs and model configs for each dataset
Flow_decomposition		-> the implementation of the Time Series Decomposition Module
High_Freq				-> the implementation of the Seasonal part
Low_Freq				-> the implementation of the Trend part
~~~



### 2. Requirements

~~~bash
pip install -r requirements.txt
~~~



### 3. Data Preparation

#### 3.1 Download Data

* The NYC dataset is derived from a fully publicly available real dataset, and the BJ dataset is derived from a map query trajectory dataset provided by a third-party location service provider.

* You can download the NYC dataset and the POIs dataset from https://data.cityofnewyork.us.

* They need to be placed under the `data` folder.

#### 3.2 Data Process

```bash
python load_train_data.py
```



### 4. Training the DeCau Model

~~~bash
python training.py
~~~

