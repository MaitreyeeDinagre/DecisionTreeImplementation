# Import Pandas and Numpy
import numpy as np
import pandas as pd

# load the data in dataframe 
oldCar = "Data_Train - Copy.xlsx"  

oldcarTrain = pd.read_excel(oldCar)

# create new column with brand name 
new = oldcarTrain["Name"].str.split(" ", n = 3, expand = True)
oldcarTrain["Brand"]= new[0]
# Update name (Ex. Audi A4 2.0  instead of Audi A4 2.0 TDI)
oldcarTrain['Name'] = new[1]+new[2]

# encode name 
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
fit = lb_make.fit_transform(oldcarTrain["Name"])
oldcarTrain['Name'] = fit
oldcarTrain.head()

# Convert Mileage, Engine and Power into int by removing KMPL,CC and BHP from end
Mileage = oldcarTrain["Mileage"].str.split(" ", n = 1, expand = True)
oldcarTrain["Mileage"]= Mileage[0]
Engine = oldcarTrain["Engine"].str.split(" ", n = 1, expand = True)
oldcarTrain["Engine"]= Engine[0]
Power = oldcarTrain["Power"].str.split(" ", n = 1, expand = True)
oldcarTrain["Power"]= Power[0]

# Convert Categorial variable to numerical level
cleanup_nums = {"Transmission": {"Manual": 1, "Automatic": 2},
                "Owner_Type": {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4},
               "Fuel_Type": {"Diesel":1,"Petrol":2,"CNG":3,"LPG":4,"Electric":5},
               "Location":{"Mumbai":1,"Pune":2,"Coimbatore":3,"Hyderabad":4,"Kochi":5,"Kolkata":6,"Delhi":7,"Chennai":8,"Jaipur":9,"Bangalore":10,"Ahmedabad":11},
               "Brand": {"Maruti": 1, "Hyundai": 2,"Honda":3,"Toyota":4,"Mercedes-Benz":5,"Volkswagen":6,"Ford":7,"Mahindra":8,"BMW":9,"Audi":10,"Tata":11,"Skoda":12,"Renault":13,"Chevrolet":14,"Nissan":15,"Land":16,"Jaguar":17,"Fiat":18,"Mitsubishi":19,"Mini":20,"Volvo":21,"Porsche":22,"Jeep":23,"Datsun":24,"Force":25,"ISUZU":26,"Lamborghini":27,"Smart":28,"Bentley":29,"Isuzu":30,"Ambassador":31}}

# Replace in original data frame
oldcarTrain.replace(cleanup_nums, inplace=True)

#drop na 
oldcarTrain = oldcarTrain.dropna(axis=0, subset=['Power'])

### remove null value entries in power
oldcarTrain = oldcarTrain.drop(oldcarTrain[oldcarTrain.Power == 'null'].index)
#print(oldcarTrain.shape)
#print(oldcarTrain.dtypes)

oldcarTrain['Mileage'] = pd.to_numeric(oldcarTrain['Mileage'])
oldcarTrain['Engine'] = pd.to_numeric(oldcarTrain['Engine'])
oldcarTrain['Power'] = pd.to_numeric(oldcarTrain['Power'],errors='ignore')
#print(oldcarTrain.dtypes)

# Drop New_price as 86% entry is missing
oldcarTrain = oldcarTrain.drop(columns = ['New_Price'], axis = 1)
#print(oldcarTrain.dtypes)

# Accornig to Cooreltaion remove seats 
oldcarTrain = oldcarTrain.drop(columns = ['Location'], axis = 1)
#print(oldcarTrain.dtypes)

X = oldcarTrain.values
y = oldcarTrain['Price'].values
X

X = np.delete(X,10,axis=1)
X

# Spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# decision-tree-regression
from sklearn.tree import DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train,y_train)
regr_2.fit(X_train,y_train)

y_pred = regr_1.predict(X_test)
y_pred2 = regr_2.predict(X_test)

print("Accuracy of decision-tree-regression with 4 level:",regr_1.score(X_test,y_test))
print("Accuracy of decision-tree-regression with 5 level:",regr_2.score(X_test,y_test))


# only linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print ("linear regression test score: ", lr_test_score)


# Printing the graph
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
feature_cols= ['Name','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats','Brand']
dot_data = StringIO()
export_graphviz(regr_2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png()) 
