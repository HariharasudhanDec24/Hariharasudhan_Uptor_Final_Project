
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

full_dataset = pd.read_csv('health_details.csv')

# Check for null values column-wise
null_summary = full_dataset.isnull().sum()
#print(null_summary)
full_dataset['Mental Health Condition'] = full_dataset['Mental Health Condition'].fillna("Unknown")

label_encoder = LabelEncoder()
columns_to_encode = ['Exercise Level','Diet Type','Stress Level',
                     'Mental Health Condition']

for col in columns_to_encode:
    full_dataset[col] = label_encoder.fit_transform(full_dataset[col])

scaler = StandardScaler()
# Standardizing data before PCA
scaled_data = scaler.fit_transform(full_dataset[['Age', 'Sleep Hours',
                                                 'Work Hours per Week', 'Social Interaction Score']])
pca = PCA(n_components=2)
work_life_balance = pca.fit_transform(scaled_data)

# Assign PCA components to the full_dataset directly
full_dataset['Work_Life_Balance_1'] = work_life_balance[:, 0]
full_dataset['Work_Life_Balance_2'] = work_life_balance[:, 1]

#Variance Plot
plt.figure(figsize=(8, 5))
plt.bar([1, 2], pca.explained_variance_ratio_, color='blue')

# Labels and title
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.title("PCA Variance (2 Components)")
plt.xticks([1, 2])
plt.show()

#Selecting features from full_dataset, including PCA components
x = full_dataset[['Exercise Level','Diet Type','Stress Level','Mental Health Condition',
                  'Work_Life_Balance_1','Work_Life_Balance_2']]
y = full_dataset['Happiness Score']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

accuracy_Using_r2 = r2_score(y_test, y_prediction)
print("Random Forest R² Score:", accuracy_Using_r2)

