import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime



conn = sqlite3.connect('crime_data.db')

query = ("""
        SELECT DATEEND, CODE_DEFINED FROM crime_2024_2
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2024_1
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2023_2
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2023_1
        UNION ALL
        SELECT DATEEND, "CODE_DEFINED" FROM crime_2022_2
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2022_1
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2021_2
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2021_1
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2020_2
        UNION ALL
        SELECT DATEEND, CODE_DEFINED FROM crime_2020_1;
        """)

df = pd.read_sql_query(query, conn)
conn.close()
df['DATEEND'].dropna()


# Try parsing with both formats using `errors='coerce'` to handle bad values
parsed_1 = pd.to_datetime(df['DATEEND'], format="%Y/%m/%d %H:%M:%S%z", errors='coerce')

parsed_2 = pd.to_datetime(df['DATEEND'], format="%m/%d/%Y %H:%M:%S%z", errors='coerce')

# Combine both, preferring the first one, then fallback to the second
df['DATEEND'] = parsed_1.fillna(parsed_2)
# Drop rows where parsing completely failed
df.dropna(subset=['DATEEND'], inplace=True)

# Optional: reset index if you dropped rows
df.reset_index(drop=True, inplace=True)

df['month'] = df['DATEEND'].dt.month         # 1–12
df['day_of_week'] = df['DATEEND'].dt.weekday # 0 (Mon) to 6 (Sun)
df['day'] = df['DATEEND'].dt.day             # 1–31
df['year'] = df['DATEEND'].dt.year  

conn = sqlite3.connect('crime_data.db')
df.to_sql('ml_data', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

'''
month_df = pd.DataFrame({
    'Burglary': [],
    'MV Theft': [],
    'Robbery': [],
    'Kidnapping': [],
    'Rape': [],
    'Murder': []
})



df['CODE_DEFINED'] = df['CODE_DEFINED'].str.upper()

# Loop over each month
total_crime_counts = df['CODE_DEFINED'].value_counts()

for month in range(1, 13):
    temp = df[df['DATEEND'].dt.month == month]['CODE_DEFINED'].value_counts()
    for crime in month_df.columns:
        label = crime.upper()
        if label in temp and label in total_crime_counts:
            month_df.loc[month, crime] = temp[label] / total_crime_counts[label]
# Reset index to have a "Month" column if needed
month_df.index.name = "Month"
month_df.reset_index(inplace=True)

#df = df[((df['CODE_DEFINED'] == 'MURDER') | (df['CODE_DEFINED'] == 'RAPE') | (df['CODE_DEFINED'] == 'ROBBERY') | (df['CODE_DEFINED'] == 'KIDNAPPING') | (df['CODE_DEFINED'] == 'MV THEFT') | (df['CODE_DEFINED'] == 'BURGLARY')) & (df['DATEEND'].dt.month == 3)]
#print(df[((df['CODE_DEFINED'] == 'MURDER') | (df['CODE_DEFINED'] == 'RAPE') | (df['CODE_DEFINED'] == 'ROBBERY') | (df['CODE_DEFINED'] == 'KIDNAPPING') | (df['CODE_DEFINED'] == 'MV THEFT') | (df['CODE_DEFINED'] == 'BURGLARY')) & (df['DATEEND'].dt.month == 3)]['CODE_DEFINED'].value_counts())

crime_types = ['Burglary', 'MV Theft', 'Robbery', 'Kidnapping', 'Rape', 'Murder']


# Day of Month
dow_df = pd.DataFrame(0.0, index=range(7), columns=crime_types)
df['CODE_DEFINED'] = df['CODE_DEFINED'].str.upper()

# Get total count of each crime
total_crime_counts = df['CODE_DEFINED'].value_counts()

for day in range(7):
    temp = df[df['DATEEND'].dt.dayofweek == day]['CODE_DEFINED'].value_counts()
    for crime in dow_df.columns:
        label = crime.upper()
        if label in temp and label in total_crime_counts:
            dow_df.loc[day, crime] = temp[label] / total_crime_counts[label]

dow_df.index.name = 'DayOfWeek'
dow_df.reset_index(inplace=True)




# Day of Month
dom_df = pd.DataFrame(0.0, index=range(1, 32), columns=crime_types)

for day in range(1, 32):
    temp = df[df['DATEEND'].dt.day == day]['CODE_DEFINED'].value_counts()
    for crime in dom_df.columns:
        label = crime.upper()
        if label in temp and label in total_crime_counts:
            dom_df.loc[day, crime] = temp[label] / total_crime_counts[label]

dom_df.index.name = 'DayOfMonth'
dom_df.reset_index(inplace=True)



# Year
# Define the fixed year range
years = list(range(2020, 2025))  # [2020, 2021, 2022, 2023, 2024]

# Initialize the year_df with those years
year_df = pd.DataFrame(0.0, index=years, columns=crime_types)

for year in years:
    # Filter for current year
    temp = df[df['DATEEND'].dt.year == year]['CODE_DEFINED'].value_counts()
    for crime in crime_types:
        label = crime.upper()
        if label in temp and label in total_crime_counts:
            year_df.loc[year, crime] = temp[label] / total_crime_counts[label]

year_df.index.name = 'Year'
year_df.reset_index(inplace=True)


conn = sqlite3.connect('crime_data.db')

month_df.to_sql('monthly_crime', conn, if_exists='replace', index=False)
dow_df.to_sql('dayofweek_crime', conn, if_exists='replace', index=False)
dom_df.to_sql('dayofmonth_crime', conn, if_exists='replace', index=False)
year_df.to_sql('yearly_crime', conn, if_exists='replace', index=False)


conn.commit()
conn.close()'''