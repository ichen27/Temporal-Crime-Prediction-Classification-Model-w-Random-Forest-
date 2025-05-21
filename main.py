import datetime
import pandas as pd
import joblib


model = joblib.load('crime_model.pkl')
le = joblib.load('label_encoder.pkl')

def predict_crime_on_date(date_str):
    date = pd.to_datetime(date_str)
    features = pd.DataFrame([{
        'month': date.month,
        'day_of_week': date.weekday(),
        'day': date.day,
        'year': date.year
    }])
    pred = model.predict_proba(features)[0]
    return dict(zip(le.classes_, pred))



def main():
    print("Crime Percentage Predicter")
    while True:
        try:
            month = int(input("Enter a month (1-12): "))
            day = int(input("Enter a day (1-31): "))
            year = int(input("Enter a year (e.g., 2023): "))

            if 1 <= month <= 12 and 1 <= day <= 31 and year >= 1900:
                break 
            else:
                print("Please enter valid values for month, day, and year.")

        except ValueError:
            print("Enter only numbers.")
    date = f"{year:04d}-{month:02d}-{day:02d}"
    print(predict_crime_on_date(date))


main()