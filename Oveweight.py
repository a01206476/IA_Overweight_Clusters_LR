#examples taken from https://realpython.com/k-means-clustering-python/

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import math
import numpy as np
from sklearn import linear_model

print("Hello, hope you're feeling well. Now let's get started with your diet plan")

print("Please enter the requested information below.")
edad = int(input("Age: "))
peso = float(input("Weight in kg: "))
estaturaBMI = float(input("Height in meters: "))
estatura = estaturaBMI * 100
genero = int(input("Gender (male = 1, female = 0): "))
af = int(input("In scale of 1 to 5, how many physical activity do you do? 1 being nothing at all and 5 being high performance athlete: "))

af = af / 10

bmi = peso / estaturaBMI ** 2
print("\nYour BMI is ", bmi)

ger = 0
if(genero == 1):
    ger = (13.75 * peso) + (5 * estatura) - (6.8 * edad) + 66

else:
    ger = ((9.6 * peso) + (1.8 * estatura)) - (4.7 * edad) + 665

print("Your Required Energy Comsuption is", round(ger,1))

get = (ger * af) + (ger * 0.1) + ger
print("Your Total Energy Consumption is", round(get,1))

## Clusters
df = pd.read_csv(r'C:\Users\just2\Documents\ProyectoIA.csv', usecols=["Edad", "IMC", "Peso", "Estatura"])

df.loc[0] = [edad, peso, estaturaBMI, bmi]
# print(df[:10])
# print()


df.plot.scatter(x='Edad', y='IMC')
# df.plot.scatter(x='Peso', y='Estatura')

# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(df)
# print(scaled_features[:10])

kmeans = KMeans(
    n_clusters = 6,
    init="k-means++",
    n_init=100,
    max_iter=1000,
    random_state=90
)
# dbscan = DBSCAN(eps=0.5)

kmeans.fit(df)

# dbscan.fit(df)

# kmeans.cluster_centers_
# sc_t = df.T
# print(df)
# plt.scatter(, c = kmeans.labels_)
myCluster = kmeans.labels_[0]

kmeans.labels_[0] = 7
df.plot.scatter(x="Edad", y="IMC", c = kmeans.labels_, cmap='rainbow')
# df.plot.scatter(x="Peso", y="Estatura", c = kmeans.labels_, cmap='rainbow')
# df.plot.scatter(x="Edad", y="IMC", c = dbscan.labels_, cmap='rainbow')

# Calculate Silhoutte Score
#
# score = silhouette_score(df, kmeans.labels_, metric='euclidean')
#
# Print the score
#
# print('Silhouetter Score: %.3f' % score)

print("\nYou are in group: ", myCluster)
requeridas = 0
if(myCluster == 0):
    print("Group 0: Represents people with a very higher BMI than expected for their age, which means you need a diet plan that diminishes the amount of Calories per day, to improve your healt and avoid chronological diseases.")
    requeridas = get - 500
if(myCluster == 1):
    print("Group 1:Represents people with a very high BMI for your age, which means you need a diet plan that diminishes the amount of Calories per day, to improve your healt and avoid chronological diseases.")
    requeridas = get - 500
if(myCluster == 2):
    print("Group 2: Represents people with an adecuate BMI for your age, \nwhich means you need a diet plan that keeps the same amount of Calories per day.")
    requeridas = get
if(myCluster == 3):
    print("Group 3: Represents people with a high BMI for your age, \nwhich means you need a diet plan that diminishes the amount of Calories per day, \nto improve your healt and avoid chronological diseases.")
    requeridas = get - 500
if(myCluster == 4):
    print("Group 4: Represents people with a very high BMI for your age, \nwhich means you need a diet plan that diminishes the amount of Calories per day, \nto improve your healt and avoid chronological diseases.")
    requeridas = get - 500
if(myCluster == 5):
    print("Group 5: Represents people with an excelent BMI for your age, \nwhich means you need a diet plan that keeps the same amount of Calories per day.")
    requeridas = get
if(myCluster == 6):
    print("Group 6: Represents people with an adecuate BMI for your age, \nwhich means you need a diet plan that keeps the same amount of Calories per day.")
    requeridas = get - 500


## Linear Regression
print("\nUsing linear regression we suggest you the next diet: ")

df = pd.read_csv(
    r'C:\Users\just2\Documents\ProyectoIA.csv', 
    usecols=["Edad", "AF", "Peso", "Estatura","Genero","GER","GET", "Requeridas","Verduras","Frutas","Cereales","Proteinas","Grasas"])

attributes = ["Edad","Peso","Estatura","AF","Genero","GER","GET","Requeridas"]

df_x = df[attributes]
df_yVerduras = df["Verduras"]
df_yFrutas = df["Frutas"]
df_yCereales = df["Cereales"]
df_yProteinas = df["Proteinas"]
df_yGrasas = df["Grasas"]

# Use 30% of the data for testing
test_size = -1 * math.floor(df_yVerduras.size * 0.3)

# Split the data into training/testing sets
x_train = df_x[:test_size]
x_test = df_x[test_size:]
yVerduras_train = df_yVerduras[:test_size]
yVerduras_test = df_yVerduras[test_size:]
yFrutas_train = df_yFrutas[:test_size]
yFrutas_test = df_yFrutas[test_size:]
yCereales_train = df_yCereales[:test_size]
yCereales_test = df_yCereales[test_size:]
yProteinas_train = df_yProteinas[:test_size]
yProteinas_test = df_yProteinas[test_size:]
yGrasas_train = df_yGrasas[:test_size]
yGrasas_test = df_yGrasas[test_size:]

# Create linear regression object
regrVerduras = linear_model.LinearRegression()
regrFrutas = linear_model.LinearRegression()
regrCereales = linear_model.LinearRegression()
regrProteinas = linear_model.LinearRegression()
regrGrasas = linear_model.LinearRegression()

# Train the model
regrVerduras.fit(x_train, yVerduras_train)
regrFrutas.fit(x_train, yFrutas_train)
regrCereales.fit(x_train, yCereales_train)
regrProteinas.fit(x_train, yProteinas_train)
regrGrasas.fit(x_train, yGrasas_train)

# Make predictions using the testing set
yVerduras_pred = regrVerduras.predict(x_test)
yFrutas_pred = regrFrutas.predict(x_test)
yCereales_pred = regrCereales.predict(x_test)
yProteinas_pred = regrProteinas.predict(x_test)
yGrasas_pred = regrGrasas.predict(x_test)

# Create data frame with predictions
df_pred = pd.DataFrame([(edad, peso, estaturaBMI, af, genero, ger, get, requeridas)], columns = attributes)
Verduras = round(regrVerduras.predict(df_pred)[0])
Frutas = round(regrFrutas.predict(df_pred)[0])
Cereales = round(regrCereales.predict(df_pred)[0])
Proteinas = round(regrProteinas.predict(df_pred)[0])
Grasas = round(regrGrasas.predict(df_pred)[0])
print("\nVerduras: " + str(Verduras))
print("Frutas: " + str(Frutas))
print("Cereales: " + str(Cereales))
print("Proteinas: " + str(Proteinas))
print("Grasas: " + str(Grasas))

print("Now the noutriologist will help you to distribute your portions depending of your daily schedule. \nWith that distribution you need to check in the equivalents table what you can have for each meal.")

print("\nMODEL Verduras")
print('Coefficients: \n', regrVerduras.coef_)
print('Mean squared error: \n %.2f' % mean_squared_error(yVerduras_test, yVerduras_pred))
print('Coefficient of determination: \n %.2f' % r2_score(yVerduras_test, yVerduras_pred))
print("\nMODEL Frutas")
print('Coefficients: \n', regrFrutas.coef_)
print('Mean squared error: \n %.2f' % mean_squared_error(yFrutas_test, yFrutas_pred))
print('Coefficient of determination: \n %.2f' % r2_score(yFrutas_test, yFrutas_pred))
print("\nMODEL Cereales")
print('Coefficients: \n', regrCereales.coef_)
print('Mean squared error: \n %.2f' % mean_squared_error(yCereales_test, yCereales_pred))
print('Coefficient of determination: \n %.2f' % r2_score(yCereales_test, yCereales_pred))
print("\nMODEL Proteinas")
print('Coefficients: \n', regrProteinas.coef_)
print('Mean squared error: \n %.2f' % mean_squared_error(yProteinas_test, yProteinas_pred))
print('Coefficient of determination: \n %.2f' % r2_score(yProteinas_test, yProteinas_pred))
print("\nMODEL Grasas")
print('Coefficients: \n', regrGrasas.coef_)
print('Mean squared error: \n %.2f' % mean_squared_error(yGrasas_test, yGrasas_pred))
print('Coefficient of determination: \n %.2f' % r2_score(yGrasas_test, yGrasas_pred))

plt.show()

# df = pd.read_csv(r'C:\Users\just2\Documents\ProyectoIA.csv', usecols=["Edad", "IMC", "Peso", "Estatura"])

# print(df[:10])
# print()


# df.plot.scatter(x='Edad', y='IMC')
# # df.plot.scatter(x='Peso', y='Estatura')

# # scaler = StandardScaler()
# # scaled_features = scaler.fit_transform(df)
# # print(scaled_features[:10])

# kmeans = KMeans(
#     n_clusters = 6,
#     init="random",
#     n_init=100,
#     max_iter=300,
#     random_state=15
# )

# # dbscan = DBSCAN(eps=1.2)

# kmeans.fit(df)

# # dbscan.fit(df)

# # kmeans.cluster_centers_
# # sc_t = df.T
# # print(df)
# # plt.scatter(, c = kmeans.labels_)
# df.plot.scatter(x="Edad", y="IMC", c = kmeans.labels_, cmap='rainbow')
# # df.plot.scatter(x="Peso", y="Estatura", c = kmeans.labels_, cmap='rainbow')
# # df.plot.scatter(x="Edad", y="IMC", c = dbscan.labels_, cmap='rainbow')
# plt.show()

# # features, true_labels = make_blobs(
# #     n_samples=200,
# #     centers=3,
# #     cluster_std=2.75,
# #     random_state = 42
# # )

# # print(features[:10])