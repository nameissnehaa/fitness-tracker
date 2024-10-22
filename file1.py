import pandas as pd
dataset = pd.read_csv(r'C:\Users\sneha\Desktop\Fitness Tracker\fitness_tracker_dataset.csv')
print(dataset.head(10))
dataset.fillna(dataset.mean(),inplace=True)
input_data = dataset[['steps','distance_km','workout_type','sleep_hours','weather_conditions','active_minutes']]
output_data = dataset[['calories_burned']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
input_data_scaled=scaler.fit_transform(input_data)


from sklearn.model_selection import train_test_split
input_data_train,input_data_test,output_data_train,output_data_test=train_test_split(input_data_scaled,output_data,test_size=0.2,random_state=42)

