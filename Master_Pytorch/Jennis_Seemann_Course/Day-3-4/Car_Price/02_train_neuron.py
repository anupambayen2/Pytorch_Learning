import pandas as pd

car_data = pd.read_csv(r'/Users/anupambayen/projects/deep_learning/Master_Pytorch/Jennis_Seemann_Course/Day-3-4/Car_Price/data/used_cars.csv', low_memory=False)

price = car_data['price']
price = price.str.replace('$','')
price = price.str.replace(',','')
price = price.astype(int)



age = car_data['model_year'].max() - car_data['model_year']

milage = car_data['milage']
milage = milage.str.replace(',','')
milage = milage.str.replace(' mi.','')
milage = milage.astype(int)

print(price)
print(age)
print(milage)
