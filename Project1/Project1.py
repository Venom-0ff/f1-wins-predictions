# Authors: Rami Abu Ahmad & Stepan Kostyukov
# Date: November, 2023

###################################
# Prepare and save a work dataset #
###################################
import pandas as pd

races = pd.read_csv("data/races.csv")
drivers = pd.read_csv("data/drivers.csv")
driver_standings = pd.read_csv("data/driver_standings.csv")

# Join driver standings and drivers into single DataFrame
df = pd.merge(driver_standings[["raceId", "driverId", "position", "wins"]], drivers[["driverId", "driverRef"]], on = "driverId", how = "left")

# Get year of each season, and exclude current year's season as it's not finished yet
df = pd.merge(df, races[["raceId", "date"]], on = "raceId", how = "left")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.year < 2023]

# Get only the champion's results
df = df[(df["position"] == 1) & (df["wins"] > 0)]

# Get total number of races per each year
df["year"] = df["date"].dt.year
races_count = df.groupby("year")["raceId"].count()
races_count = races_count.reset_index().rename(columns = {"raceId": "racesCount"})

# Get only the results after the final race of the season
final_races = df.groupby("year")["date"].max()
final_races = final_races.reset_index()

df = df.merge(final_races["date"], left_on=df["date"], right_on = final_races["date"])
df = pd.merge(df, races_count, on = "year", how = "left")
df = df.sort_values(by="year")

# Save the resulting DataFrame into a separate my_dataset.csv file for easier use later on
df[["year", "driverRef", "wins", "racesCount"]].to_csv("data/my_dataset.csv")
print("DataFrame was saved into data/my_dataset.csv!")



####################
# Train the models #
####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

my_dataset = pd.read_csv("data/my_dataset.csv")

# Split the dataset into train and test parts
train_test_split_ratio = 0.8
train_size = int(len(my_dataset) * train_test_split_ratio)
train_set = my_dataset.iloc[:train_size, :]
test_set = my_dataset.iloc[train_size:, :]


x_train = train_set["racesCount"]
x_train = np.array(x_train)
y_train = train_set["wins"]

x_test = test_set["racesCount"]
x_test = np.array(x_test)
y_test = test_set["wins"]


#####################
# Linear Regression #
#####################
from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
model_reg.fit(x_train.reshape(-1, 1), y_train)

x_reg = x_test.reshape(-1, 1)
y_reg = model_reg.predict(x_reg)

# Evaluate Linear Regression
mae_reg = mean_absolute_error(y_test, y_reg)
mse_reg = mean_squared_error(y_test, y_reg)
rmse_reg = np.sqrt(mse_reg)
r2_reg = r2_score(y_test, y_reg)

print("==========================================================")
print("Evaluate Linear Regression:")
print(f"Mean Absolute Error: {mae_reg}")
print(f"Mean Squared Error: {mse_reg}")
print(f"Root Mean Squared Error: {rmse_reg}")
print(f"R-squared: {r2_reg}")
print("==========================================================")


#######
# SVR #
#######
from sklearn.svm import SVR
model_svr = SVR(kernel="linear")
model_svr.fit(x_train.reshape(-1, 1), y_train)

x_svr = x_test.reshape(-1, 1)
y_svr = model_svr.predict(x_svr)

# Evaluate SVR
mae_svr = mean_absolute_error(y_test, y_svr)
mse_svr = mean_squared_error(y_test, y_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_svr)

print("Evaluate SVR:")
print(f"Mean Absolute Error: {mae_svr}")
print(f"Mean Squared Error: {mse_svr}")
print(f"Root Mean Squared Error: {rmse_svr}")
print(f"R-squared: {r2_svr}")
print("==========================================================")


##########################
# Neural Network (Keras) #
##########################
from keras.models import Sequential
from keras.layers import Dense

model_nn = Sequential()
model_nn.add(Dense(64, input_dim=1, activation="relu"))
model_nn.add(Dense(32, activation="relu"))
model_nn.add(Dense(1, activation="linear"))
model_nn.compile(optimizer="adam", loss="mean_squared_error")
model_nn.fit(x_train, y_train, epochs=1000, verbose=0)

x_nn = x_test
y_nn = model_nn.predict(x_nn)

# Evaluate NN
mae_nn = mean_absolute_error(y_test, y_nn)
mse_nn = mean_squared_error(y_test, y_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_nn)

print("==========================================================")
print("Evaluate Neural Network:")
print(f"Mean Absolute Error: {mae_nn}")
print(f"Mean Squared Error: {mse_nn}")
print(f"Root Mean Squared Error: {rmse_nn}")
print(f"R-squared: {r2_nn}")
print("==========================================================")


##################
# Visualize Data #
##################
historic_mean = my_dataset.groupby("racesCount")["wins"].mean().reset_index()

plt.axes().set_facecolor("gray")
plt.grid(True)
plt.scatter(my_dataset["racesCount"], my_dataset["wins"], label="Historic Data")
plt.plot(historic_mean["racesCount"], historic_mean["wins"], color = "cyan", label="Historic Mean")
plt.scatter(x_reg, y_reg, color="yellow", label="Predicted Data (Linear Regression)")
plt.scatter(x_svr, y_svr, color="red", label="Predicted Data (SVR)")
plt.scatter(x_nn, y_nn, color="blue", label="Predicted Data (Neural Networks)")
plt.xlabel("racesCount")
plt.ylabel("wins")
plt.legend()
plt.show()