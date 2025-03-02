import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

def rank_city_for_holidays(cityname , df ):
    city_df = df[df['City'].str.lower() == cityname.lower()].copy()
    if city_df.empty:
        return f"No travel destinations found for {cityname}."
    
    features = ['Google review rating', 'Number of google review in lakhs', 'Entrance Fee in INR', 'time needed to visit in hrs']

    weights = {
        'Google review rating': 0.4,
        'Number of google review in lakhs': 0.3,
        'Entrance Fee in INR': 0.2,
        'time_needed_scaled': 0.1  
    }
    scaler = MinMaxScaler()
    city_df[['Google review rating', 'Number of google review in lakhs']] = scaler.fit_transform(city_df[['Google review rating', 'Number of google review in lakhs']])
    city_df['Entrance Fee in INR'] = 1 - scaler.fit_transform(city_df[['Entrance Fee in INR']])  
    city_df['time_needed_scaled'] = 1 - scaler.fit_transform(city_df[['time needed to visit in hrs']])

    city_df['score'] = (
        weights['Google review rating'] * city_df['Google review rating'] +
        weights['Number of google review in lakhs'] * city_df['Number of google review in lakhs'] +
        weights['Entrance Fee in INR'] * city_df['Entrance Fee in INR'] +
        weights['time_needed_scaled'] * city_df['time_needed_scaled']
    )

    city_df['rank'] = city_df['score'].rank(ascending=False, method='min')
    city_df = city_df.sort_values('rank')

    return city_df[['Name', 'City', 'Google review rating', 'Number of google review in lakhs', 'Entrance Fee in INR', 'rank']].head(10)

if __name__ == '__main__':
    file_path = './archive/dataset.csv'
    df = pd.read_csv(file_path)
    city = input("Enter your city name:\n")
    result = rank_city_for_holidays(city, df)
    print (result)