
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# === Cargar datos ===
df = pd.read_csv("C:/ProyectoBigDta/Modelo/kidney_disease.csv")
df.drop('id', axis=1, inplace=True)

# === Renombrar columnas ===
df.columns = ['edad', 'presion_sanguinea', 'gravedad_especifica', 'albumina', 'azucar',
              'globulos_rojos', 'celulas_pus', 'grumos_celulas_pus', 'bacterias',
              'glucosa_aleatoria', 'urea_sanguinea', 'creatinina_serica', 'sodio',
              'potasio', 'hemoglobina', 'volumen_celular_empacado',
              'recuento_globulos_blancos', 'recuento_globulos_rojos',
              'hipertension', 'diabetes_mellitus', 'enfermedad_arterial_coronaria',
              'apetito', 'edema_pedal', 'anemia', 'clase']

# === Limpieza específica de texto/tabulaciones ===
df['diabetes_mellitus'].replace({' yes': 'yes', '\tyes': 'yes', '\tno': 'no'}, inplace=True, regex=True)
df['enfermedad_arterial_coronaria'].replace({'\tno': 'no'}, inplace=True, regex=True)
df['clase'].replace({'ckd\t': 'ckd'}, inplace=True, regex=True)

# === Conversión de columnas numéricas con errores ===
df['volumen_celular_empacado'] = pd.to_numeric(df['volumen_celular_empacado'], errors='coerce')
df['recuento_globulos_blancos'] = pd.to_numeric(df['recuento_globulos_blancos'], errors='coerce')
df['recuento_globulos_rojos'] = pd.to_numeric(df['recuento_globulos_rojos'], errors='coerce')

# === Codificación manual de variables categóricas ===
bin_map = {
    'normal': 1, 'abnormal': 0,
    'present': 1, 'notpresent': 0,
    'yes': 1, 'no': 0,
    'good': 1, 'poor': 0,
    'ckd': 0, 'notckd': 1
}

df['globulos_rojos'] = df['globulos_rojos'].map({'normal': 1, 'abnormal': 0})
df['celulas_pus'] = df['celulas_pus'].map({'normal': 1, 'abnormal': 0})
df['grumos_celulas_pus'] = df['grumos_celulas_pus'].map({'present': 1, 'notpresent': 0})
df['bacterias'] = df['bacterias'].map({'present': 1, 'notpresent': 0})
df['hipertension'] = df['hipertension'].map({'yes': 1, 'no': 0})
df['diabetes_mellitus'] = df['diabetes_mellitus'].map({'yes': 1, 'no': 0})
df['enfermedad_arterial_coronaria'] = df['enfermedad_arterial_coronaria'].map({'yes': 1, 'no': 0})
df['apetito'] = df['apetito'].map({'good': 1, 'poor': 0})
df['edema_pedal'] = df['edema_pedal'].map({'yes': 1, 'no': 0})
df['anemia'] = df['anemia'].map({'yes': 1, 'no': 0})
df['clase'] = df['clase'].map({'ckd': 0, 'notckd': 1})

# === Rellenar valores faltantes ===
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# === División ===
X = df.drop(columns=['clase'])
y = df['clase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# === Entrenamiento ===
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

modelo.fit(X_train, y_train)

# === Evaluación ===
y_pred = modelo.predict(X_test)
print("Exactitud:", accuracy_score(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

# === Guardar modelo ===
with open("C:/ProyectoBigDta/Modelo/modelo_entrenado.pkl", "wb") as f:
    pickle.dump(modelo, f)


print("Modelo guardado como modelo_entrenado.pkl")
