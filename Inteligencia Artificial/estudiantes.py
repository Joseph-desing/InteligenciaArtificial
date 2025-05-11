import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Creamos un conjunto de datos artificial
datos = {
    'HorasEstudio': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ConocimientoPrevio': [2, 3, 4, 3, 5, 6, 7, 8, 9, 10],
    'Asistencia': [50, 60, 55, 65, 70, 80, 85, 90, 95, 100],
    'PromedioTareas': [60, 65, 66, 70, 75, 78, 80, 85, 90, 95],
    'TipoEstudiante': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    'Resultado': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(datos)
X = df.drop('Resultado', axis=1)  
y = df['Resultado']               

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print("Precisión del modelo:", precision)
# Mostrar predicciones
print("\nPredicciones del modelo:")
for i in range(len(X_test)):
    print(f"Entrada: {X_test.iloc[i].to_dict()} → Predicción: {'Aprobado' if predicciones[i] == 1 else 'No Aprobado'}")
