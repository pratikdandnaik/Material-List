from google.colab import drive
drive.mount('/content/drive')
pip install tensorflow --upgrade
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
df = pd.read_csv('/content/drive/MyDrive/Thesis/Data.csv')
print(df.head(4))
df['Material'] = df[['Std', 'Material', 'Heat treatment']].fillna('').agg(' '.join, axis=1)
df['Sy'] = df['Sy'].str.replace(' max', '').astype(int)
df.drop(['Std', 'ID', 'Heat treatment', 'Desc', 'A5', 'Bhn', 'pH', 'Desc', 'HV'], axis=1, inplace=True)
print(df.head(4))
df.info()

df['Use'] = (
    (df['Su'].between(292, 683)) &
    (df['Sy'].between(212, 494)) &
    (df['E'].between(196650, 217350)) &
    (df['G'].between(47400, 110600)) &
    (df['mu'].between(0.225, 0.375)) &
    (df['Ro'].between(6288, 9432))
).map({True: 'Yes', False: 'No'})

df.insert(1, 'Use', df.pop('Use'))
print(df.head(4))
sy_sum = df['Sy'].sum()
su_sum = df['Su'].sum()

# Values for the pie chart
values = [sy_sum, su_sum]
labels = ['Yield Strength (Sy)', 'Ultimate Tensile Strength (Su)']

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of Yield Strength (Sy) and Ultimate Tensile Strength (Su)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
g_sum = df['G'].sum()
ro_sum = df['Ro'].sum()
e_sum = df['E'].sum()
values = [g_sum, ro_sum, e_sum]
labels = ['Shear Modulus (G)', 'Density (Ro)', 'Elastic Modulus (E)']
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'],
                                   wedgeprops=dict(width=0.3))

# Add a circle at the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Shear Modulus (G), Density (Ro), and Elastic Modulus (E)')
plt.axis('equal')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(df['Su'], kde=True, bins=30)
plt.title('Histogram of Ultimate Tensile Strength (Su)')
plt.xlabel('Ultimate Tensile Strength (Su) in MPa')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['Sy'])
plt.title('Box Plot of Yield Strength (Sy)')
plt.ylabel('Yield Strength (Sy) in MPa')
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Su', y='Sy', hue='Use', data=df)
plt.title('Scatter Plot of Ultimate Tensile Strength (Su) vs. Yield Strength (Sy)')
plt.xlabel('Ultimate Tensile Strength (Su) in MPa')
plt.ylabel('Yield Strength (Sy) in MPa')
plt.legend(title='Use')
plt.show()
plt.figure(figsize=(8, 6))
corr_matrix = df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
sns.pairplot(df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro', 'Use']], hue='Use', diag_kind='kde', palette='Set2')
plt.suptitle('Pair Plot of Mechanical Properties', y=1.02)
plt.show()
# Prepare the data for the  model
X = df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro']].values
y = df['Use'].map({'Yes': 1, 'No': 0}).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for feature extraction
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca = X_train_pca.reshape(X_train_pca.shape[0], X_train_pca.shape[1], 1)
X_test_pca = X_test_pca.reshape(X_test_pca.shape[0], X_test_pca.shape[1], 1)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True)
}

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


results = {}
for name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)



best_model = models['Random Forest']
best_model.fit(X_train, y_train)
df['Rating'] = best_model.predict_proba(scaler.transform(df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro']].values))[:, 1]
df['Predicted_Use'] = df['Rating'].apply(lambda x: 'Yes' if x > 0.5 else 'No')


df['s.no'] = df.index + 1


output_df = df[['s.no', 'Material', 'Rating', 'Predicted_Use']]


top_5_materials = output_df.sort_values(by='Rating', ascending=False).head(5)

# Display the top 5 materials
print("Top 5 materials for chassis use:")
print(top_5_materials)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with additional metrics
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='f1_score', curve='PR')])

model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
X_all = scaler.transform(df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro']].values)
X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)
# Predict and assign the 'Rating' and 'Predicted_Use' columns
df['Rating'] = model.predict(X_all).flatten()
df['Predicted_Use'] = df['Rating'].apply(lambda x: 'Yes' if x > 0.5 else 'No')


df['s.no'] = df.index + 1
output_df = df[['s.no', 'Material', 'Rating', 'Predicted_Use']]


print(output_df.head(10))

top_5_materials = df.sort_values(by='Rating', ascending=False).head(5)

# Display the top 5 materials
print("Top 5 materials for chassis use:")
print(top_5_materials)
from tensorflow.keras.layers import GRU, Dense
model = Sequential()
model.add(GRU(64, activation='tanh', input_shape=(X_train_pca.shape[1], 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(32, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with additional metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='f1_score', curve='PR')])

model.summary()
history = model.fit(X_train_pca, y_train, epochs=100, batch_size=32, validation_split=0.2)
eval_results = model.evaluate(X_test_pca, y_test)
print(f'Accuracy: {eval_results[1]:.2f}')
print(f'Precision: {eval_results[2]:.2f}')
print(f'Recall: {eval_results[3]:.2f}')
print(f'F1 Score: {eval_results[4]:.2f}')

