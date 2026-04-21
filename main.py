from ucimlrepo import fetch_ucirepo 
from preprossing import preprocess
from model import train
from evulate import evaluate_model
# fetch dataset 
metro_interstate_traffic_volume = fetch_ucirepo(id=492) 

lineSperator = '-' * 50

# data (as pandas dataframes) 
X = metro_interstate_traffic_volume.data.features 
y = metro_interstate_traffic_volume.data.targets 
  
# metadata 
print(metro_interstate_traffic_volume.metadata) 
print(lineSperator)
  
# variable information 
print(metro_interstate_traffic_volume.variables) 
print(lineSperator)

# Preprocess
X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()

# train model

y_pred = train(X_train,y_train,X_test)

#Evaluate results
evaluate_model(y_pred, X_test, y_test)


  
