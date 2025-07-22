### Clean the data from the source, I.E. removing unneeded data, reorganizing columns, and combining into a single file
### Dataset source: https://www.kaggle.com/datasets/unsdsn/world-happiness/data

import pandas as pd
from pathlib import Path

data_folder = Path(__file__).parent

years = range(2015, 2020)

file_paths = [data_folder / f"data/{year}.csv" for year in years]

dataframes = []
for year in years:
    file_path = data_folder / f"data/{year}.csv"
    df = pd.read_csv(file_path)
    df['Year'] = year  # Add a 'Year' column
    
    if year == 2015:
        df = df.drop('Standard Error', axis=1)
        df = df.drop('Region', axis=1)
        df = df.drop('Dystopia Residual', axis=1)
        df = df.drop(columns=['Happiness Rank'])

    elif year == 2016:
        df = df.drop('Lower Confidence Interval', axis=1)
        df = df.drop('Upper Confidence Interval', axis=1)
        df = df.drop('Region', axis=1)
        df = df.drop('Dystopia Residual', axis=1)
        df = df.drop(columns=['Happiness Rank'])

    elif year == 2017:
        df = df.drop('Whisker.high', axis=1)
        df = df.drop('Whisker.low', axis=1)
        df = df.drop('Dystopia.Residual', axis=1)
        df['Temp'] = df['Generosity']
        df['Generosity'] = df['Trust..Government.Corruption.']
        df['Trust..Government.Corruption.'] = df['Temp']
        df = df.drop(columns=['Temp'])
        df = df.drop(columns=['Happiness.Rank'])

    elif year == 2018:
        df['Temp'] = df['Generosity']
        df['Generosity'] = df['Perceptions of corruption']
        df['Perceptions of corruption'] = df['Temp']
        df = df.drop(columns=['Temp'])
        df = df.drop(columns=['Overall rank'])

    elif year == 2019:
        df['Temp'] = df['Generosity']
        df['Generosity'] = df['Perceptions of corruption']
        df['Perceptions of corruption'] = df['Temp']
        df = df.drop(columns=['Temp'])
        df = df.drop(columns=['Overall rank'])

    df.columns = ['Country','Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','year']

    if year != 2019:
        dataframes.append(df)
    else:
        df.to_csv(path_or_buf=data_folder / f"data/2019_actual.csv", index=False)
        df = df.drop('Happiness Score', axis=1) # Drop happiness score so that model will have to predict it
        df.to_csv(path_or_buf=data_folder / f"data/2019_holdout.csv", index=False)
    print(f"{year} added")
    
df = pd.concat(dataframes, ignore_index=True)
df.to_csv(path_or_buf=data_folder / f"data/data.csv", index=False)
print("Files cleaned and compiled under data/data.csv!")
