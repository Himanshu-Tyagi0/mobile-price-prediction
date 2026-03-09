import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")

# load dataset
df = pd.read_csv(r"C:\Users\tyagi\Downloads\mobile_price_5000.csv")

print("Dataset preview")
print(df.head())

# encode display type
df["display_type"] = df["display_type"].map({"IPS":0,"AMOLED":1})

# remove outliers using IQR
Q1 = df["selling_price"].quantile(0.25)
Q3 = df["selling_price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["selling_price"] >= lower) & (df["selling_price"] <= upper)]

# drop unnecessary column
df = df.drop("brand",axis=1)

# define features and target
X = df.drop("selling_price",axis=1)
y = df["selling_price"]

# train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# create models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
dt = DecisionTreeRegressor(random_state=42)

# train models
lr.fit(X_train,y_train)
rf.fit(X_train,y_train)
dt.fit(X_train,y_train)

# predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)

# evaluation function
def evaluate(name,y_test,pred):

    r2 = r2_score(y_test,pred)
    mae = mean_absolute_error(y_test,pred)
    mse = mean_squared_error(y_test,pred)
    rmse = np.sqrt(mse)

    print("\nModel:",name)
    print("R2:",r2)
    print("MAE:",mae)
    print("MSE:",mse)
    print("RMSE:",rmse)

    return r2


# evaluate models
lr_r2 = evaluate("Linear Regression",y_test,lr_pred)
rf_r2 = evaluate("Random Forest",y_test,rf_pred)
dt_r2 = evaluate("Decision Tree",y_test,dt_pred)

# model comparison
scores = {
    "Linear Regression":lr_r2,
    "Random Forest":rf_r2,
    "Decision Tree":dt_r2
}

best_model_name = max(scores,key=scores.get)

print("\nBest model:",best_model_name)

if best_model_name == "Linear Regression":
    best_model = lr

elif best_model_name == "Random Forest":
    best_model = rf

else:
    best_model = dt

# actual vs predicted plot
pred = best_model.predict(X_test)

plt.figure(figsize=(7,6))
sns.scatterplot(x=y_test,y=pred,color="royalblue",alpha=0.7)

sns.regplot(x=y_test,y=pred,scatter=False,color="red")

plt.xlabel("Actual Price",fontsize=12)
plt.ylabel("Predicted Price",fontsize=12)
plt.title("Actual vs Predicted Selling Price",fontsize=14)
plt.show()

# residual plot
residuals = y_test - pred

plt.figure(figsize=(7,6))
sns.scatterplot(x=pred,y=residuals,color="darkorange",alpha=0.7)

plt.axhline(0,color="black",linestyle="--")

plt.xlabel("Predicted Price",fontsize=12)
plt.ylabel("Residual Error",fontsize=12)
plt.title("Residual Error Plot",fontsize=14)

plt.show()

# feature importance
feature_names = X.columns
importance = best_model.coef_
importance_pct = importance / importance.sum() * 100
plt.figure(figsize=(14,6))
plt.bar(feature_names, importance_pct,color=["blue", "red", "green", "orange"])
plt.title("Which Feature Matters Most?")
plt.xlabel("Feature")
plt.ylabel("Importance (%)")
plt.show()

# predict new phone price
new_phone = pd.DataFrame({

    "actual_price":[65000],
    "age_years":[2],
    "ram_gb":[8],
    "storage_gb":[128],
    "battery_mah":[5000],
    "display_type":[1],
    "camera_mp":[50],
    "scratches_on_lens":[0],
    "dent_on_body":[0]

})

predicted_price = best_model.predict(new_phone)

print("\nNew Phone Details")
print(new_phone)

print("\nPredicted Selling Price:",predicted_price[0])