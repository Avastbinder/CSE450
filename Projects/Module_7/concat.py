import pandas as pd
from pathlib import Path

# Set your data folder (adjust if needed)
data_folder = Path.cwd()  # Or Path("path/to/your/folder")

# List of years you want to load
years = range(2015, 2020)

# Create list of file paths
file_paths = [data_folder / f"Projects/Module_7/data/{year}.csv" for year in years]

# Create a list of DataFrames with Year column attached
dataframes = []
for year in years:
    file_path = data_folder / f"Projects/Module_7/data/{year}.csv"
    df = pd.read_csv(file_path)
    df['Year'] = year  # Add a 'Year' column
    cols = list(df.columns)
    #i, j = cols.index('A'), cols.index('C')
    #cols[i], cols[j] = cols[j], cols[i]
    if year == '2015':
        df = df.drop('Standard Error', axis=1)
        df = df.drop('Region', axis=1)
        df = df.drop('Dystopia Residual', axis=1)
    elif year == '2016':
        df = df.drop('Lower Confidence Interval', axis=1)
        df = df.drop('Upper Confidence Interval', axis=1)
        df = df.drop('Region', axis=1)
        df = df.drop('Dystopia Residual', axis=1)
    elif year == '2017':
        df = df.drop('Whisker.high', axis=1)
        df = df.drop('Whisker.low', axis=1)
        df = df.drop('Dystopia Residual', axis=1)
        cols[cols.index('Generosity')], cols[cols.index('Trust..Government.Corruption.')] = cols[cols.index('Trust..Government.Corruption.')], cols[cols.index('Generosity')] # swap columns
    elif year == '2018':
        cols[cols.index('Generosity')], cols[cols.index('Trust..Government.Corruption.')] = cols[cols.index('Trust..Government.Corruption.')], cols[cols.index('Generosity')] # swap columns
    elif year == '2019':
        \
    df.columns = ['Country','Region','Happiness Rank','Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual','year']
    dataframes.append(df)
    
df = pd.concat(dataframes, ignore_index=True)

df.to_csv(path_or_buf=f"Projects/Module_7/data/data.csv", index=False)

