import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import sqlalchemy as sa
from sqlalchemy import create_engine

# =============================================
# CONFIGURACIÓN INICIAL Y CARGA DE DATOS
# =============================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Análisis de Salud del Sueño - Dataset"
server = app.server  

# Asegurar que exista la carpeta de assets
if not os.path.exists('assets'):
    os.makedirs('assets')

# Conexión a PostgreSQL
try:
    # Configuración de la base de datos
    DB_USER = 'postgres'
    DB_PASSWORD = 'molly5011'
    DB_HOST = 'localhost'
    DB_PORT = '5432'
    DB_NAME = 'sleepdb'
    
    # Crear conexión a PostgreSQL
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    
    # Cargar datos desde PostgreSQL
    data_base = pd.read_sql('SELECT * FROM sleep_data', engine)

    # Estandarizar nombres de columnas
    data_base.columns = [col.replace(' ', '_') for col in data_base.columns]

    # Preprocesamiento
    data_base['Sleep_Disorder'] = data_base['Sleep_Disorder'].fillna('None')

    
except Exception as e:
    print(f"Error al conectar con PostgreSQL: {e}")
    
    # Como fallback, cargar desde CSV local
    try:
        # Definir rutas relativas para los archivos
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "Sleep_health_and_lifestyle_dataset.csv")
        
        data_base = pd.read_csv(data_path)
        
        # Limpiar nombres de columnas (quitar espacios)
        data_base.columns = [col.replace(' ', '_') for col in data_base.columns]
        
        # Preprocesamiento
        data_base['Sleep_Disorder'] = data_base['Sleep_Disorder'].fillna('None')
        
    except Exception as e:
        print(f"Error al cargar datos desde CSV: {e}")
        # Crear datos ficticios para evitar errores si el archivo no está disponible
        data_base = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Age': [35, 42, 29, 51],
            'Sleep_Duration': [7.5, 6.8, 8.2, 6.5],
            'Quality_of_Sleep': [8, 6, 9, 5],
            'Physical_Activity_Level': [60, 45, 75, 30],
            'Stress_Level': [5, 7, 4, 8],
            'BMI_Category': ['Normal', 'Overweight', 'Normal', 'Obese'],
            'Heart_Rate': [72, 78, 68, 82],
            'Daily_Steps': [8500, 6000, 10000, 5000],
            'Sleep_Disorder': ['None', 'Insomnia', 'None', 'Sleep Apnea']
        })

# Preparar datos para correlaciones
numeric_vars = [
    'Sleep_Duration', 'Quality_of_Sleep', 
    'Physical_Activity_Level', 'Stress_Level', 
    'Heart_Rate', 'Daily_Steps'
]

# Calcular matriz de correlación
corr_matrix = data_base[numeric_vars].corr()

# =============================================
# PROCESAMIENTO DE DATOS
# =============================================

# Calcular estadísticas por grupo
stats_gender = data_base.groupby('Gender')[numeric_vars].mean().reset_index()
stats_bmi = data_base.groupby('BMI_Category')[numeric_vars].mean().reset_index()

# Preparar datos para gráficas
gender_counts = data_base['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

bmi_counts = data_base['BMI_Category'].value_counts().reset_index()
bmi_counts.columns = ['BMI_Category', 'Count']

sleep_disorder_counts = data_base['Sleep_Disorder'].value_counts().reset_index()
sleep_disorder_counts.columns = ['Sleep_Disorder', 'Count']

# Estadísticas descriptivas
sleep_stats = data_base['Sleep_Duration'].describe().reset_index()
sleep_stats.columns = ['Estadística', 'Valor']

# =============================================
# MODELADO PREDICTIVO
# =============================================

# Crear una función para entrenar el modelo y obtener resultados
def train_model():
    # Seleccion de variables características y objetivo
    X = data_base.drop(['Sleep_Disorder', 'Person_ID', 'Occupation', 'Blood_Pressure'], axis=1, errors='ignore')
    y = data_base['Sleep_Disorder']
    
    # Definir columnas por tipo
    numeric_features = ['Age', 'Sleep_Duration', 'Quality_of_Sleep',
                        'Physical_Activity_Level', 'Stress_Level',
                        'Heart_Rate', 'Daily_Steps']
    categorical_features = ['Gender', 'BMI_Category']
    
    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42, stratify=y)
    
    # MODELO: Random Forest con balanceo de clases
    rf_model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('balancer', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100,
                                             max_depth=10,
                                             min_samples_split=5,
                                             random_state=42))
    ])
    
    # Entrenamiento
    rf_model.fit(X_train, y_train)
    
    # Validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='balanced_accuracy')
    
    # Predicciones
    y_pred = rf_model.predict(X_test)
    
    # Extraer importancia de características
    rf_classifier = rf_model.named_steps['classifier']
    importances = rf_classifier.feature_importances_
    
    # Obtener nombres de características después de la transformación
    cat_features = rf_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_features)
    
    # Calcular matriz de confusión y métricas
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': rf_model,
        'feature_names': feature_names,
        'feature_importances': importances,
        'X_train': X_train, 
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'cv_scores': cv_scores,
        'confusion_matrix': cm,
        'report': classification_report(y_test, y_pred, output_dict=True),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro')
    }

# Entrenar modelo y obtener resultados
model_results = train_model()

# Preparar importancia de características para visualización
feature_importance_df = pd.DataFrame({
    'Feature': model_results['feature_names'],
    'Importance': model_results['feature_importances']
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# =============================================
# COMPONENTES VISUALES
# =============================================

# Definir colores para la aplicación
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'lightgray': '#ecf0f1',
    'chart1': '#1E90FF',
    'chart2': '#FF69B4',
    'chart3': '#32CD32',
    'chart4': '#FFA500',
    'chart5': '#9370DB',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Definir estilo de tarjeta para elementos
card_style = {
    'backgroundColor': '#e8eaf6',
    'borderRadius': '10px',
    'padding': '15px',
    'marginBottom': '15px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
}

# Navbar
navbar = dbc.NavbarSimple(
    brand="💡 ANÁLISIS Y CLASIFICACIÓN DE LOS TRASTORNOS DEL SUEÑO 🧠",
    brand_href="#",
    color="primary",
    dark=True,
    brand_style={
        "fontWeight": "1000",         # más grueso que "bold"
        "fontSize": "35px",          # tamaño más grande
        "letterSpacing": "1px",      # más espaciado
        "textTransform": "uppercase" # todo en mayúscula por consistencia
    }
)



# Tarjetas informativas
def create_card(title, value, color, prefix="", suffix=""):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.H2(f"{prefix}{value}{suffix}", className="card-text")
        ]),
        color=color,
        inverse=True,
        className="mb-3"
    )

cards = dbc.Row([
    dbc.Col(create_card("Duración Promedio de Sueño", f"{data_base['Sleep_Duration'].mean():.2f}", "success", suffix=" horas")),
    dbc.Col(create_card("Calidad Promedio de Sueño", f"{data_base['Quality_of_Sleep'].mean():.2f}", "info", suffix="/10")),
    dbc.Col(create_card("Nivel Promedio de Estrés", f"{data_base['Stress_Level'].mean():.2f}", "warning", suffix="/10"))
])

# =============================================
# PESTAÑAS DEL DASHBOARD
# =============================================

# 1. Introducción
introduccion_tab = dcc.Tab(
    label='1. Introducción',
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("ANÁLISIS DE SALUD DEL SUEÑO", className="mb-2", style={'fontWeight': 'bold'}),
                        html.H5("Eliana Fuentes & María Camila Vargas", className="mb-4", style={'color': colors['primary']}),
                        html.P(
                            """
                            El presente dashboard interactivo tiene como propósito analizar los patrones de sueño, estilo de vida y posibles trastornos
                            del sueño en una muestra de individuos, basándose en el conjunto de datos 'Sleep Health and Lifestyle'. La información contenida
                            permite identificar relaciones entre variables demográficas, fisiológicas y conductuales que influyen en la calidad y duración del sueño.
                            """,
                            className="mb-3"
                        ),
                        html.P(
                            """
                            Además del análisis exploratorio, se ha desarrollado un modelo predictivo basado en Random Forest que permite clasificar con precisión los trastornos del sueño más comunes,
                            como el insomnio y la apnea. Esta herramienta tiene aplicaciones potenciales en el ámbito de la salud pública, la medicina preventiva y el bienestar general.
                            """,
                            className="mb-3"
                        ),
                        html.P(
                            """
                            La estructura de este dashboard sigue un enfoque metodológico riguroso que va desde la contextualización y el planteamiento del problema,
                            hasta los resultados del modelo y sus implicaciones prácticas. Esperamos que este análisis proporcione una visión clara, intuitiva y útil sobre los factores que impactan en la salud del sueño.
                            """
                        )
                    ], style={
                        'backgroundColor': '#eaf2f8',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.05)'
                    })
                ], md=8),
                dbc.Col([
                    html.Img(
                        src="/assets/intro_sleep.png",
                        style={'height': '400px', 'marginTop': '20px'}
                    )
                ], md=4, style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
            ], className="mt-4 mb-4"),
            html.Hr(),
            cards
        ], fluid=True)
    ]
)


# 2. Contexto
contexto_tab = dcc.Tab(
    label='2. Contexto',
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="/assets/context_variables.png",
                        style={'width': '100%', 'maxWidth': '600px', 'height': 'auto', 'margin': 'auto'}
                    )
                ], md=4, style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                dbc.Col([
                    html.H2("Contextualización del Dataset", className="mb-4"),
                    html.P("Fuente: Sleep Health and Lifestyle Dataset"),
                    html.P("Registros: 400 individuos  Actualización: Marzo 2025"),
                    html.H4("Variables Clave:"),
                    html.Ul([
                        html.Li([
                            html.Span("Person ID", style={'color': '#e75480'}), ": Un identificador único para cada individuo."
                        ]),
                        html.Li([
                            html.Span("Gender", style={'color': '#e75480'}), ": El género de la persona (Masculino/Femenino)."
                        ]),
                        html.Li([
                            html.Span("Age", style={'color': '#e75480'}), ": La edad de la persona en años."
                        ]),
                        html.Li([
                            html.Span("Occupation", style={'color': '#e75480'}), ": La ocupación o profesión de la persona."
                        ]),
                        html.Li([
                            html.Span("Sleep Duration (hours)", style={'color': '#e75480'}), ": El número de horas que la persona duerme al día."
                        ]),
                        html.Li([
                            html.Span("Quality of Sleep (scale: 1-10)", style={'color': '#e75480'}), ": Una calificación subjetiva de la calidad del sueño."
                        ]),
                        html.Li([
                            html.Span("Physical Activity Level (minutes/day)", style={'color': '#e75480'}), ": El número de minutos que la persona dedica a la actividad física diaria."
                        ]),
                        html.Li([
                            html.Span("Stress Level (scale: 1-10)", style={'color': '#e75480'}), ": Una calificación subjetiva del nivel de estrés experimentado por la persona."
                        ]),
                        html.Li([
                            html.Span("BMI Category", style={'color': '#e75480'}), ": La categoría de IMC de la persona (Bajo peso, Normal, Sobrepeso, Obesidad)."
                        ]),
                        html.Li([
                            html.Span("Blood Pressure", style={'color': '#e75480'}), ": La medición de la presión arterial."
                        ]),
                        html.Li([
                            html.Span("Heart Rate (bpm)", style={'color': '#e75480'}), ": La frecuencia cardíaca en reposo en latidos por minuto."
                        ]),
                        html.Li([
                            html.Span("Daily Steps", style={'color': '#e75480'}), ": El número de pasos que la persona camina al día."
                        ]),
                        html.Li([
                            html.Span("Sleep Disorder", style={'color': '#e75480'}), ": La presencia o ausencia de un trastorno del sueño (Ninguno, Insomnio, Apnea del sueño)."
                        ])
                    ])
                ], md=8)
            ], className="mb-5"),

            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.H3("Distribución por Género", className="mb-4 mt-4"),
                    dcc.Graph(
                        figure=px.pie(
                            gender_counts,
                            values='Count',
                            names='Gender',
                            color='Gender',
                            color_discrete_map={'Male': colors['chart1'], 'Female': colors['chart2']},
                            hole=0.4
                        ).update_layout(
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background'],
                            margin=dict(l=20, r=20, t=30, b=20),
                        )
                    )
                ], md=4),
                dbc.Col([
                    html.H3("Distribución por Categoría de IMC", className="mb-4 mt-4"),
                    dcc.Graph(
                        figure=px.pie(
                            bmi_counts,
                            values='Count',
                            names='BMI_Category',
                            color_discrete_sequence=[colors['chart1'], colors['chart2'], colors['chart3'], colors['chart4']],
                            hole=0.4
                        ).update_layout(
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background'],
                            margin=dict(l=20, r=20, t=30, b=20),
                        )
                    )
                ], md=4),
                dbc.Col([
                    html.H3("Distribución por Trastorno del Sueño", className="mb-4 mt-4"),
                    dcc.Graph(
                        figure=px.pie(
                            sleep_disorder_counts,
                            values='Count',
                            names='Sleep_Disorder',
                            color_discrete_sequence=[colors['chart1'], colors['chart2'], colors['chart3']],
                            hole=0.4
                        ).update_layout(
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background'],
                            margin=dict(l=20, r=20, t=30, b=20),
                        )
                    )
                ], md=4)
            ])
        ], fluid=True)
    ]
)



# 3. Planteamiento del Problema
problema_tab = dcc.Tab(
    label='3. Problema',
    children=[
        dbc.Container([
            html.H2(
                "🛌 PLANTEAMIENTO DEL PROBLEMA 😴",
                className="mt-4 mb-5 text-center",
                style={
                    'fontWeight': 'bold',
                    'color': '#1e5aa8',
                    'fontSize': '2.3rem',
                    'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                    'borderBottom': f'3px solid {colors["primary"]}',
                    'display': 'inline-block',
                    'paddingBottom': '10px'
                }
            ),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🔍 Problema Propuesto", className="mb-3"),
                        html.H5("Predicción y clasificación de trastornos del sueño basados en factores de estilo de vida y salud",
                                className="mb-4", style={'color': colors['primary']}),

                        html.P("Este análisis busca construir un modelo predictivo que permita identificar y clasificar la presencia de trastornos del sueño (como insomnio o apnea) a partir de variables conductuales y fisiológicas registradas en el conjunto de datos.", style={'textAlign': 'justify'}),

                        html.Ul([
                            html.Li("Clasificar el tipo de trastorno del sueño presente en cada individuo."),
                            html.Li("Determinar qué variables tienen mayor influencia en la aparición de estos trastornos."),
                            html.Li("Apoyar con herramientas analíticas la prevención y monitoreo de la salud del sueño.")
                        ]),

                        html.H4("❓ Pregunta Problema", className="mb-3 mt-4"),
                        html.Blockquote([
                            html.P("¿Es posible predecir la presencia y tipo de trastorno del sueño en una persona a partir de sus hábitos de sueño, nivel de actividad física, estrés y estado de salud general?")
                        ], className="blockquote", style={
                            'borderLeft': f'5px solid {colors["primary"]}',
                            'paddingLeft': '15px',
                            'backgroundColor': '#f9f9f9',
                            'borderRadius': '8px'
                        })

                    ], style=card_style)
                ], md=8),

                dbc.Col([
                    html.Img(src="assets/problem_sleep.png", style={'width': '100%', 'borderRadius': '12px'})
                ], md=4)
            ])
        ], fluid=True)
    ]
)



# 4. Objetivos y Justificación
objetivos_tab = dcc.Tab(
    label='4. Objetivos',
    children=[
        dbc.Container([
            html.H2(
                "🎯 OBJETIVOS Y JUSTIFICACIÓN 🧠",
                className="mt-4 mb-5 text-center",
                style={
                    'fontWeight': 'bold',
                    'color': '#1e5aa8',
                    'fontSize': '2.5rem',
                    'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                    'borderBottom': f'3px solid {colors["primary"]}',
                    'display': 'inline-block',
                    'paddingBottom': '10px'
                }
            ),

            # Objetivo general + específicos al lado de la imagen
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🎯 Objetivo General", className="mb-3"),
                        html.P("Desarrollar un modelo predictivo que permita clasificar con precisión los trastornos del sueño (None, Insomnia, Sleep Apnea) a partir de variables relacionadas con el estilo de vida y la salud.")
                    ], style=card_style),

                    html.Div([
                        html.H4("📌 Objetivos Específicos", className="mb-3 mt-4"),
                        html.Ul([
                            html.Li("Identificar las variables de mayor influencia en la predicción de cada tipo de trastorno del sueño."),
                            html.Li("Evaluar el desempeño del modelo en términos de precisión, recall y F1-score para cada clase."),
                            html.Li("Desarrollar perfiles de riesgo para cada tipo de trastorno del sueño basados en los patrones identificados."),
                            html.Li("Proponer recomendaciones personalizadas basadas en los factores modificables más relevantes para cada tipo de trastorno.")
                        ])
                    ], style=card_style)
                ], md=8),

                dbc.Col([
                    html.Img(src="assets/objetivos_illustration.png", style={'width': '100%', 'borderRadius': '12px'})
                ], md=4)
            ], className="mb-5"),

            # Justificación
            html.Div([
                html.H4("🧠 Justificación", className="mb-3 mt-4"),
                html.P("Los trastornos del sueño representan un problema de salud pública significativo con impactos en múltiples dimensiones:"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Impacto en la Salud", className="bg-primary text-white"),
                            dbc.CardBody([
                                html.P("Los trastornos del sueño están asociados con mayor riesgo de enfermedades cardiovasculares, diabetes, obesidad y problemas de salud mental.")
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Impacto Económico", className="bg-info text-white"),
                            dbc.CardBody([
                                html.P("Reducción de productividad, mayor ausentismo laboral y aumento en costos de atención médica.")
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Calidad de Vida", className="bg-success text-white"),
                            dbc.CardBody([
                                html.P("Deterioro en la calidad de vida, relaciones sociales y bienestar general de las personas afectadas.")
                            ])
                        ])
                    ], md=4)
                ], className="mt-3")
            ], style=card_style)
        ], fluid=True)
    ]
)



# 5. Marco Teórico
marco_tab = dcc.Tab(
    label='5. Marco Teórico',
    children=[
        dbc.Container([
            html.H2(
                "📚 MARCO TEÓRICO 🧾",
                className="mt-4 mb-5 text-center",
                style={
                    'fontWeight': 'bold',
                    'color': '#1e5aa8',
                    'fontSize': '2.5rem',
                    'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                    'borderBottom': f'3px solid {colors["primary"]}',
                    'display': 'inline-block',
                    'paddingBottom': '10px'
                }
            ),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Trastornos del sueño", className="mb-3", style={'color': colors['primary'], 'fontWeight': 'bold'}),
                        html.P("Se refieren a alteraciones en la cantidad, calidad o ritmo del sueño que afectan la salud."),
                        html.H5("• Insomnio", className="mt-3", style={'color': colors['chart2']}),
                        html.P("Dificultad para dormir o mantener el sueño. Puede ser crónico si ocurre al menos 3 veces por semana durante 3 meses."),
                        html.H5("• Apnea del sueño", className="mt-3", style={'color': colors['chart2']}),
                        html.P("Interrupciones en la respiración durante el sueño, asociadas con fatiga diurna y problemas de salud graves."),

                        html.H4("Factores que afectan la calidad del sueño", className="mt-5 mb-4", style={'color': colors['primary'], 'fontWeight': 'bold'}),
                        dbc.Row([
                            dbc.Col([
                                html.H5("🧬 Fisiológicos", className="mb-2", style={'color': colors['chart1']}),
                                html.Ul([
                                    html.Li("Edad: Cambios naturales en los ciclos de sueño."),
                                    html.Li("Género: Diferencias hormonales afectan el sueño."),
                                    html.Li("IMC: Relación directa con apnea del sueño.")
                                ])
                            ], md=4),

                            dbc.Col([
                                html.H5("🏃 Conductuales", className="mb-2", style={'color': colors['chart1']}),
                                html.Ul([
                                    html.Li("Actividad física: Mejora la calidad del sueño."),
                                    html.Li("Consumo de sustancias: Puede alterar el descanso."),
                                    html.Li("Rutinas: Dormir a diferentes horas afecta negativamente.")
                                ])
                            ], md=4),

                            dbc.Col([
                                html.H5("🧠 Psicológicos", className="mb-2", style={'color': colors['chart1']}),
                                html.Ul([
                                    html.Li("Estrés: Inhibe el inicio del sueño."),
                                    html.Li("Ansiedad: Provoca despertares frecuentes."),
                                    html.Li("Depresión: Relacionada con insomnio o exceso de sueño.")
                                ])
                            ], md=4)
                        ])
                    ], style=card_style)
                ], md=8),

                dbc.Col([
                    html.Img(src="assets/theory_sleep.png", style={'width': '100%', 'borderRadius': '12px'})
                ], md=4)
            ], className="mb-5"),

            html.Div([
                html.H4("Aplicación del Aprendizaje Automático en Trastornos del Sueño", className="mb-4", style={'color': colors['primary'], 'fontWeight': 'bold'}),
                html.P(
                    "Los trastornos del sueño representan un reto clínico importante en salud pública. Modelos de machine learning como Random Forest han demostrado ser "
                    "efectivos para identificar patrones complejos entre variables fisiológicas y conductuales, facilitando su diagnóstico temprano. "
                    "Su capacidad para manejar datos heterogéneos y ofrecer interpretabilidad lo convierten en un aliado ideal para este tipo de análisis predictivo."
                ),
                html.P(
                    "En el presente proyecto se utiliza Random Forest para predecir la clase de trastorno del sueño a partir de datos como edad, duración del sueño, IMC, nivel de estrés, "
                    "actividad física, frecuencia cardíaca, entre otros. La selección de este modelo se fundamenta no solo en su desempeño empírico, sino también en su validación "
                    "en la literatura reciente."
                ),
                html.H5("📚 Soporte en Literatura Reciente", className="mt-4 mb-3", style={'color': colors['chart1']}),
                html.P("A continuación se resumen algunos trabajos relevantes que sustentan este enfoque:"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🔬 Torres et al. (2023)", style={'backgroundColor': colors['primary'], 'color': 'white'}),
                            dbc.CardBody([
                                html.P(
                                    "Desarrollaron una app móvil que implementa un modelo de machine learning para detectar apnea del sueño, "
                                    "a partir de señales respiratorias y polisomnografía. El modelo se proyecta como herramienta clínica efectiva.",
                                    className="card-text"
                                ),
                                html.A("Ver estudio", href="https://hdl.handle.net/20.500.12442/12851", target="_blank")
                            ])
                        ])
                    ], md=4),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("👶 Pacho Velasco (2022)", style={'backgroundColor': colors['chart3'], 'color': 'white'}),
                            dbc.CardBody([
                                html.P(
                                    "Aplicó Random Forest para detectar spindles del sueño en niños con apnea, logrando un 96.51% de precisión. "
                                    "El estudio evidenció los retos del desbalance de clases y la importancia del preprocesamiento.",
                                    className="card-text"
                                ),
                                html.A("Ver estudio", href="https://uvadoc.uva.es/bitstream/handle/10324/57405/TFG-G5868.pdf", target="_blank")
                            ])
                        ])
                    ], md=4),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🧠 Satyam & Chandra (2025)", style={'backgroundColor': colors['chart1'], 'color': 'white'}),
                            dbc.CardBody([
                                html.P(
                                    "Compararon distintos algoritmos de clasificación aplicados a trastornos del sueño. Random Forest destacó "
                                    "por su precisión y adaptabilidad en conjuntos de datos clínicos multivariados.",
                                    className="card-text"
                                ),
                                html.A("Ver estudio", href="https://www.irjmets.com/uploadedfiles/paper/issue_4_april_2025/74444/final/fin_irjmets1746339611.pdf", target="_blank")
                            ])
                        ])
                    ], md=4)
                ])
            ], style=card_style)
        ], fluid=True)
    ]
)




# 6. Metodología con subtabs mejorada con imágenes y contenido extendido
metodologia_subtabs = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('📌 DEFINICIÓN DEL PROBLEMA', style={'color': colors['primary'], 'fontWeight': 'bold'}),
                    html.P("En este proyecto se aborda un problema de clasificación multiclase cuyo objetivo principal es predecir si una persona presenta un trastorno del sueño (insomnio, apnea o ninguno), utilizando variables relacionadas con salud y estilo de vida."),
                    html.Ul([
                        html.Li("Tipo de problema: Clasificación multiclase"),
                        html.Li("Variable objetivo: Sleep_Disorder (None, Insomnia, Sleep Apnea)"),
                        html.Li("Justificación: Identificar tempranamente estos trastornos mejora la calidad de vida y permite una intervención oportuna.")
                    ])
                ], style=card_style)
            ], md=8),
            dbc.Col([
                html.Img(src="assets/problem_definition.png", style={'width': '60%', 'borderRadius': '12px'})
            ], md=4)
        ])
    ]),

    dcc.Tab(label='b. Preparación de los Datos', children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('🧹 PREPARACIÓN DE LOS DATOS', style={'color': colors['primary'], 'fontWeight': 'bold'}),
                    html.H5("Limpieza y transformación de datos:", style={'color': '#d9534f'}),
                    html.Ul([
                        html.Li("Tratamiento de valores nulos en 'Sleep_Disorder' reemplazándolos por 'None'."),
                        html.Li("Estandarización de nombres de columnas para asegurar consistencia."),
                        html.Li("Eliminación de variables irrelevantes como IDs o presión arterial cruda."),
                        html.Li("Escalado de variables numéricas para mejorar el rendimiento del modelo."),
                        html.Li("Codificación one-hot para convertir variables categóricas en formato numérico.")
                    ]),
                    html.H5("Balanceo y división:", style={'color': '#d9534f'}),
                    html.Ul([
                        html.Li("Aplicación de SMOTE para mitigar el desbalance de clases."),
                        html.Li("Separación 75%-25% entre entrenamiento y prueba usando estratificación."),
                        html.Li("Uso de validación cruzada estratificada para asegurar generalización.")
                    ])
                ], style=card_style)
            ], md=8),
            dbc.Col([
                html.Img(src="assets/data_preparation.png", style={'width': '80%', 'borderRadius': '12px'})
            ], md=4)
        ])
    ]),

    dcc.Tab(label='c. Selección del Modelo', children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('🤖 SELECCIÓN DEL MODELO', style={'color': colors['primary'], 'fontWeight': 'bold'}),
                    html.P("Se probaron dos modelos de clasificación: Ridge Classifier y Random Forest Classifier."),
                    html.H5("Modelo elegido: Random Forest", style={'color': '#d9534f'}),
                    html.Ul([
                        html.Li("Mejor desempeño en recall ponderado."),
                        html.Li("Manejo efectivo del desbalance."),
                        html.Li("Capacidad de interpretar la importancia de variables.")
                    ]),
                    html.H5("Representación matemática:", style={'color': colors['primary']}),
                    dcc.Markdown('''
                        $$\hat{y} = \text{modo}(h_1(x), h_2(x), \dots, h_n(x))$$

                        Donde $h_i(x)$ representa la predicción de la clase por parte del árbol $i$ en el bosque aleatorio,
                        y $\hat{y}$ es la clase más votada (voto mayoritario). Esta representación refleja el principio
                        del Random Forest, que combina múltiples árboles de decisión para mejorar la precisión.
                    ''', mathjax=True)
                ], style=card_style)
            ], md=8),
            dbc.Col([
                html.Img(src="assets/model_selection.png", style={'width': '80%', 'borderRadius': '12px'})
            ], md=4)
        ])
    ]),

    dcc.Tab(label='d. Entrenamiento y Evaluación', children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('📊 ENTRENAMIENTO Y EVALUACIÓN DEL MODELO', style={'color': colors['primary'], 'fontWeight': 'bold'}),
                    html.P("El modelo fue entrenado con datos transformados y balanceados. La evaluación se realizó usando validación cruzada estratificada (5 folds), lo cual garantiza una mejor estimación del rendimiento al mantener la distribución de clases en cada partición."),
                    html.H5("📈 Métrica principal seleccionada: Recall ponderado (Weighted Recall)", style={'color': '#d9534f'}),
                    html.P("Esta métrica refleja la capacidad del modelo para capturar correctamente los casos positivos en todas las clases, ponderando según su frecuencia. Es esencial en contextos clínicos, ya que ayuda a minimizar los falsos negativos."),
                    html.Ul([
                        html.Li("Balanced Recall CV: 0.83 ± 0.02"),
                        html.Li("F1-score, Precision y Recall por clase en conjunto de prueba")
                    ])
                ], style=card_style)
            ], md=8),
            dbc.Col([
                html.Img(src="assets/model_training.png", style={'width': '85%', 'borderRadius': '12px'})
            ], md=4)
        ])
    ])
])

# Encapsular en la pestaña principal 6
metodologia_tab = dcc.Tab(
    label='6. Metodología',
    children=[
        dbc.Container([
            html.H2("🧪 METODOLOGÍA 🧬", className="mt-4 mb-5 text-center", style={
                'fontWeight': 'bold',
                'color': '#1e5aa8',
                'fontSize': '2.5rem',
                'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                'borderBottom': f'3px solid {colors["primary"]}',
                'display': 'inline-block',
                'paddingBottom': '10px'
            }),
            metodologia_subtabs
        ], fluid=True)
    ]
)




# 7. Resultados y Análisis Final con subtabs
resultados_subtabs = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('Análisis Exploratorio de Datos (EDA)', className="mt-4 mb-3"),
        html.Div([
            cards,  # Tarjetas con métricas clave
            html.Hr(),
            html.H5("Distribuciones de variables clave", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=px.histogram(
                            data_base, 
                            x='Sleep_Duration',
                            title='Distribución de la Duración del Sueño',
                            color_discrete_sequence=[colors['chart1']]
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background']
                        )
                    )
                ], md=6),
                dbc.Col([
                    dcc.Graph(
                        figure=px.histogram(
                            data_base, 
                            x='Quality_of_Sleep',
                            title='Distribución de la Calidad del Sueño',
                            color_discrete_sequence=[colors['chart2']]
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background']
                        )
                    )
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=px.histogram(
                            data_base, 
                            x='Stress_Level',
                            title='Distribución del Nivel de Estrés',
                            color_discrete_sequence=[colors['chart3']]
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background']
                        )
                    )
                ], md=6),
                dbc.Col([
                    dcc.Graph(
                        figure=px.histogram(
                            data_base, 
                            x='Physical_Activity_Level',
                            title='Distribución del Nivel de Actividad Física',
                            color_discrete_sequence=[colors['chart4']]
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background']
                        )
                    )
                ], md=6)
            ]),
            html.Hr(),
            html.H5("Matriz de Correlación", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=px.imshow(
                            corr_matrix,
                            text_auto=True,
                            zmin=-1,
                            zmax=1,
                            color_continuous_scale='RdBu_r',
                            aspect='equal',
                            title='Correlación entre Variables Numéricas'
                        ).update_layout(
                            height=600,
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background'],
                            margin=dict(l=50, r=50, t=100, b=50),
                            coloraxis_colorbar_title='Correlación'
                        )
                    )
                ], md=12)
            ])
        ])
    ]),
    dcc.Tab(label='b. EDA 2', children=[
        html.H4('Análisis por Perfiles', className="mt-4 mb-3"),
        html.Div([
            html.H5("Perfiles por Género", className="mt-3 mb-3"),
            # Tabla resumen por género
            html.Div([
                html.H6("Estadísticas por Género", className="mt-3 mb-3 text-center"),
                dash_table.DataTable(
                    columns=[
                        {"name": "Género", "id": "Gender"},
                        {"name": "Duración Sueño (h)", "id": "Sleep_Duration", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Calidad Sueño", "id": "Quality_of_Sleep", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Actividad Física (min)", "id": "Physical_Activity_Level", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Nivel Estrés", "id": "Stress_Level", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    data=stats_gender.to_dict('records'),
                    style_header={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 0},
                            'backgroundColor': 'rgba(255, 105, 180, 0.1)'
                        },
                        {
                            'if': {'row_index': 1},
                            'backgroundColor': 'rgba(30, 144, 255, 0.1)'
                        }
                    ],
                )
            ], style={'marginBottom': '30px'}),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=px.box(
                            data_base,
                            x='Gender',
                            y='Sleep_Duration',
                            color='Gender',
                            title='Duración del Sueño por Género',
                            color_discrete_map={'Male': colors['chart1'], 'Female': colors['chart2']}
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background']
                        )
                    )
                ], md=6),
                dbc.Col([
                    dcc.Graph(
                        figure=px.box(
                            data_base,
                            x='Gender',
                            y='Quality_of_Sleep',
                            color='Gender',
                            title='Calidad del Sueño por Género',
                            color_discrete_map={'Male': colors['chart1'], 'Female': colors['chart2']}
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background']
                        )
                    )
                ], md=6)
            ]),
            html.Hr(),
            html.H5("Perfiles por IMC", className="mt-4 mb-3"),
            # Tabla resumen por IMC
            html.Div([
                html.H6("Estadísticas por Categoría de IMC", className="mt-3 mb-3 text-center"),
                dash_table.DataTable(
                    columns=[
                        {"name": "Categoría IMC", "id": "BMI_Category"},
                        {"name": "Duración Sueño (h)", "id": "Sleep_Duration", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Calidad Sueño", "id": "Quality_of_Sleep", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Actividad Física (min)", "id": "Physical_Activity_Level", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Nivel Estrés", "id": "Stress_Level", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    data=stats_bmi.to_dict('records'),
                    style_header={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgba(52, 152, 219, 0.1)'
                        }
                    ],
                )
            ], style={'marginBottom': '30px'}),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=px.histogram(
                            data_base,
                            x='BMI_Category',
                            color='Sleep_Disorder',
                            barmode='group',
                            title='Distribución de Trastornos del Sueño por Categoría de IMC',
                            color_discrete_sequence=[colors['chart1'], colors['chart2'], colors['chart3']]
                        ).update_layout(
                            template='plotly_white',
                            plot_bgcolor=colors['background'],
                            paper_bgcolor=colors['background'],
                            xaxis_title='Categoría de IMC',
                            yaxis_title='Conteo',
                            legend_title='Trastorno del Sueño'
                        )
                    )
                ], md=12)
            ])
        ])
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('Visualización de Resultados del Modelo', className="mt-4 mb-3"),
        html.Div([
            html.H5("Importancia de las Características", className="mt-3 mb-3"),
            dcc.Graph(
                figure=px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Importancia de Características en la Predicción de Trastornos del Sueño',
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Blues
                ).update_layout(
                    template='plotly_white',
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    yaxis={'categoryorder': 'total ascending'}
                )
            ),
            
            html.Div([
                html.H5("Factores Predictivos de Trastornos del Sueño", className="mt-4 mb-3"),
                html.P("La importancia de características revela que los principales factores predictivos son:"),
                html.Ul([
                    html.Li([html.B("BMI_Category_Overweight"), ": El sobrepeso aparece como el factor más importante (0.21)"]),
                    html.Li([html.B("Physical_Activity_Level"), ": El nivel de actividad física es el segundo factor (0.17)"]),
                    html.Li([html.B("Age"), ": La edad es el tercer factor más importante (0.14)"]),
                    html.Li([html.B("Sleep_Duration"), ": La duración del sueño (0.12)"]),
                    html.Li([html.B("Daily_Steps"), ": Los pasos diarios (0.11)"])
                ]),
                html.P("Esto sugiere que el manejo del peso y el nivel de actividad física son cruciales para prevenir trastornos del sueño.")
            ], className="mt-4 mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #e9ecef'}),
            
            html.Hr(),
            html.H5("Matriz de Confusión", className="mt-4 mb-3"),
            dcc.Graph(
                figure=px.imshow(
                    model_results['confusion_matrix'],
                    labels=dict(x="Predicción", y="Real", color="Conteo"),
                    x=model_results['model'].classes_,
                    y=model_results['model'].classes_,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Confusión - Random Forest",
                    color_continuous_scale='Blues'
                ).update_layout(
                    template='plotly_white',
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background']
                )
            ),
            
            html.Div([
                html.H5("Evaluación del Modelo: Random Forest", className="mt-4 mb-3"),
                html.P("El modelo de Random Forest fue evaluado usando múltiples métricas con enfoque en problemas multiclase y desbalanceados. Se obtuvo el siguiente desempeño:"),
                html.Ul([
                    html.Li([html.B("Recall ponderado (0.9255): "), "esta métrica fue priorizada dado que nos interesa minimizar los falsos negativos en cada categoría, especialmente para no omitir pacientes con trastornos como Insomnio o Apnea del Sueño."]),
                    html.Li([html.B("Balanced Accuracy (0.9240): "), "útil en contextos de desbalance, ya que promedia el recall por clase, compensando la predominancia de la clase 'None'."]),
                    html.Li([html.B("F1 Macro (0.9069): "), "ofrece una medida equilibrada entre precisión y recall para cada clase, permitiendo comparar el rendimiento global."])
                ]),
                html.H6("Informe de Clasificación:", className="mt-3 mb-2"),
                html.Ul([
                    html.Li([html.B("Insomnia: "), "Recall de 0.89 y F1-score de 0.89 → el modelo tiene buen desempeño identificando correctamente esta clase."]),
                    html.Li([html.B("None: "), "Recall de 0.93 → identifica muy bien a los pacientes sin trastornos, aunque hubo 4 falsos positivos (3 Sleep Apnea, 1 Insomnia)."]),
                    html.Li([html.B("Sleep Apnea: "), "Recall de 0.95, aunque con menor precisión (0.79), indicando que algunos pacientes fueron mal clasificados como apnea."])
                ]),
                html.H6("Conclusión:", className="mt-3 mb-2"),
                html.P("El modelo logra una excelente capacidad de generalización, como lo demuestran las métricas similares entre entrenamiento y prueba. Dado el enfoque clínico del problema, se priorizó el recall ponderado para evitar falsos negativos, y el modelo respondió adecuadamente a esta necesidad. La matriz de confusión refuerza esto, mostrando pocos errores graves de clasificación. El recall ponderado alto es fundamental para un problema sensible como este, en el que dejar pasar un trastorno sin detectar puede tener consecuencias clínicas relevantes.")
            ], className="mt-4 mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #e9ecef'}),
            
            html.Hr(),
            html.H5("Comparación de Desempeño para Evaluar Overfitting", className="mt-4 mb-3"),
            html.Div([
                dcc.Markdown('''
    
                ''', className="mb-3"),
                
                html.Div([
                    html.H6("Evaluación de Overfitting", className="mt-3 mb-2"),
                    html.P("Se compararon las métricas de desempeño entre el conjunto de entrenamiento y el conjunto de prueba para detectar posibles signos de sobreajuste (overfitting)."),
                    html.P("Los resultados fueron los siguientes:"),
                    html.Ul([
                        html.Li([html.B("Recall ponderado (Train): "), "0.9179"]),
                        html.Li([html.B("Recall ponderado (Test): "), "0.9255"]),
                        html.Li([html.B("F1 ponderado (Train): "), "0.9173"]),
                        html.Li([html.B("F1 ponderado (Test): "), "0.9276"])
                    ]),
                    html.P([
                        "Dado que los valores en ambos conjuntos son ", 
                        html.B("muy similares"), 
                        " e incluso ligeramente superiores en el conjunto de prueba, se concluye que el modelo ",
                        html.B("no presenta overfitting"), "."
                    ]),
                    html.P([
                        "Esto indica una ",
                        html.B("buena capacidad de generalización"),
                        ", lo cual es crucial al aplicar el modelo a nuevos datos."
                    ])
                ], className="mt-3 p-3", style={'backgroundColor': '#e8f4f8', 'borderRadius': '5px', 'border': '1px solid #bde0ec'})
            ], className="mt-4 mb-4"),
            
            html.Hr(),
            html.H5("Identificación de Predicciones Incorrectas", className="mt-4 mb-3"),
            html.Div([
                dcc.Markdown('''
    
                ''', className="mb-3"),
                
                # Gráfico PCA (imagen cargada desde assets)
                html.Div([
                    html.Img(src="assets/pca.png", alt="Visualización PCA de predicciones", 
                            style={'maxWidth': '100%', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
                    html.Figcaption("Visualización de predicciones en espacio PCA", 
                                style={'textAlign': 'center', 'fontStyle': 'italic', 'marginTop': '10px'})
                ], className="text-center mb-4"),
                
                html.Div([
                    html.H6("Interpretación del Análisis PCA", className="mt-3 mb-2"),
                    html.P("La proyección del conjunto de prueba sobre los dos primeros componentes principales (PCA) permite visualizar la distribución de las observaciones en un espacio reducido de 2 dimensiones, facilitando la evaluación visual del comportamiento del modelo."),
                    html.P("En el gráfico:"),
                    html.Ul([
                        html.Li("Cada punto representa una instancia del conjunto de prueba."),
                        html.Li([
                            "El ", 
                            html.B("color"), 
                            " indica la clase ", 
                            html.B("predicha"), 
                            " por el modelo (", 
                            html.Code("None"), 
                            ", ", 
                            html.Code("Insomnia"), 
                            ", ", 
                            html.Code("Sleep Apnea"), 
                            ")."
                        ]),
                        html.Li([
                            "La ", 
                            html.B("forma"), 
                            " del punto distingue si la predicción fue correcta (✔️) o incorrecta (✖️)."
                        ])
                    ]),
                    html.H6("Interpretación:", className="mt-3 mb-2"),
                    html.Ul([
                        html.Li([
                            "Se observan ", 
                            html.B("grupos bien definidos"), 
                            " en el espacio PCA, lo cual indica que el modelo logró capturar estructuras discriminativas entre clases."
                        ]),
                        html.Li("La mayoría de los puntos están correctamente clasificados (predicción = real), lo que concuerda con el bajo porcentaje de errores observado (solo 7.4% de predicciones incorrectas)."),
                        html.Li("Las pocas instancias mal clasificadas aparecen aisladas o en zonas de frontera entre clases, lo cual es esperable en problemas multiclase con cierto traslape.")
                    ]),
                    html.P("Esta visualización complementa las métricas cuantitativas al permitir validar gráficamente la capacidad de generalización del modelo y detectar regiones donde podría haber ambigüedad entre clases.")
                ], className="mt-3 p-3", style={'backgroundColor': '#eef7ee', 'borderRadius': '5px', 'border': '1px solid #c3e6cb'})
            ], className="mt-4 mb-4")
        ])
    ]),
    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('Indicadores de Evaluación del Modelo', className="mt-4 mb-3"),
        html.Div([
            html.H5("Métricas de Rendimiento por Clase", className="mt-3 mb-3"),
            html.Div([
                dash_table.DataTable(
                    columns=[
                        {"name": "Clase", "id": "class"},
                        {"name": "Precisión", "id": "precision", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "Recall", "id": "recall", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "F1-Score", "id": "f1-score", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "Support", "id": "support", "type": "numeric"}
                    ],
                    data=[
                        {
                            "class": cls,
                            "precision": model_results['report'][cls]['precision'],
                            "recall": model_results['report'][cls]['recall'],
                            "f1-score": model_results['report'][cls]['f1-score'],
                            "support": model_results['report'][cls]['support']
                        } for cls in model_results['report'] if cls not in ['accuracy', 'macro avg', 'weighted avg']
                    ],
                    style_header={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px'
                    }
                )
            ], style={'marginBottom': '30px'}),
            html.Hr(),
            html.H5("Métricas Globales", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    create_card("Balanced Accuracy", f"{model_results['balanced_accuracy']:.4f}", "primary")
                ], md=4),
                dbc.Col([
                    create_card("Macro F1-Score", f"{model_results['macro_f1']:.4f}", "success")
                ], md=4),
                dbc.Col([
                    create_card("Precisión Promedio", f"{model_results['report']['weighted avg']['precision']:.4f}", "info")
                ], md=4)
            ]),
            html.Hr(),
            html.H5("Interpretación de los Resultados", className="mt-4 mb-3"),
            html.Ul([
                html.Li([
                    html.Strong("Balanced Accuracy: "), 
                    f"{model_results['balanced_accuracy']:.4f} - El modelo tiene un buen rendimiento equilibrado en todas las clases."
                ]),
                html.Li([
                    html.Strong("F1-Score: "), 
                    f"El F1-Score macro de {model_results['macro_f1']:.4f} indica un buen equilibrio entre precisión y recall."
                ]),
                html.Li([
                    html.Strong("Por clase: "), 
                    "El modelo muestra mejor rendimiento en la detección de 'None' y 'Sleep Apnea' que en la detección de 'Insomnia'."
                ])
            ])
        ])
    ]),
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('Limitaciones y Consideraciones Finales', className="mt-4 mb-3"),
        html.Div([
            html.H5("Limitaciones del Análisis", className="mt-3 mb-3"),
            html.Ul([
                html.Li("Tamaño de la muestra limitado (400 individuos), lo que puede afectar la generalización del modelo."),
                html.Li("Posible presencia de variables no medidas que podrían influir en los trastornos del sueño (como medicamentos, condiciones médicas específicas, factores genéticos)."),
                html.Li("Datos subjetivos para algunas variables clave (calidad del sueño, nivel de estrés) que pueden introducir sesgos."),
                html.Li("Desbalance en las clases de la variable objetivo, que aunque se corrigió con SMOTE, puede afectar el rendimiento real del modelo.")
            ]),
            html.Hr(),
            html.H5("Posibles Mejoras", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Mejoras en Datos", className="bg-primary text-white"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Ampliar el tamaño de la muestra"),
                                html.Li("Incluir más variables relevantes (consumo de cafeína, alcohol, calidad del ambiente de sueño)"),
                                html.Li("Mediciones objetivas de calidad del sueño (polisomnografía)")
                            ], className="mb-0")
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Mejoras en Modelado", className="bg-success text-white"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Probar otros algoritmos (XGBoost, SVM, Redes Neuronales)"),
                                html.Li("Optimización de hiperparámetros más exhaustiva"),
                                html.Li("Experimentar con otras técnicas de balanceo de clases")
                            ], className="mb-0")
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Mejoras en Análisis", className="bg-info text-white"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Análisis de subgrupos específicos"),
                                html.Li("Incorporación de análisis longitudinal"),
                                html.Li("Validación externa con otros conjuntos de datos")
                            ], className="mb-0")
                        ])
                    ])
                ], md=4)
            ])
        ])
    ])
])

resultados_tab = dcc.Tab(
    label='7. Resultados', 
    children=[
        dbc.Container([
            html.H2('Resultados y Análisis Final', className="mt-4 mb-4"),
            resultados_subtabs
        ], fluid=True)
    ]
)


# 8. Conclusiones
conclusiones_tab = dcc.Tab(
    label='8. Conclusiones', 
    children=[
        dbc.Container([
            html.H2('Conclusiones', className="mt-4 mb-4"),
            html.Div([
                html.H4("Hallazgos Principales", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Diferencias de Género", className="mb-0"), 
                                         style={'backgroundColor': colors['primary'], 'color': 'white'}),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li(f"Las mujeres duermen más (promedio {stats_gender.loc[stats_gender['Gender'] == 'Female', 'Sleep_Duration'].values[0]:.2f} horas) que los hombres ({stats_gender.loc[stats_gender['Gender'] == 'Male', 'Sleep_Duration'].values[0]:.2f} horas)."),
                                    html.Li(f"La calidad del sueño también es diferente: {stats_gender.loc[stats_gender['Gender'] == 'Female', 'Quality_of_Sleep'].values[0]:.2f}/10 en mujeres vs {stats_gender.loc[stats_gender['Gender'] == 'Male', 'Quality_of_Sleep'].values[0]:.2f}/10 en hombres.")
                                ])
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Factores Predictivos", className="mb-0"), 
                                         style={'backgroundColor': colors['secondary'], 'color': 'white'}),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li(f"Factores más importantes para predecir trastornos del sueño: {', '.join(feature_importance_df['Feature'].iloc[:3].tolist())}"),
                                    html.Li("El IMC es un fuerte predictor para la apnea del sueño, mientras que el estrés lo es para el insomnio.")
                                ])
                            ])
                        ])
                    ], md=6)
                ], className="mb-4"),
                
                html.H4("Relevancia de los Resultados", className="mb-3 mt-4"),
                html.Ul([
                    html.Li("El modelo desarrollado logra predecir con buena precisión los trastornos del sueño, especialmente la apnea del sueño."),
                    html.Li("La identificación de factores determinantes permite diseñar intervenciones específicas para cada tipo de trastorno."),
                    html.Li("El análisis confirma relaciones conocidas (como IMC y apnea del sueño) y revela nuevas asociaciones que pueden guiar futuras investigaciones.")
                ]),
                
                html.H4("Recomendaciones Basadas en el Análisis", className="mb-3 mt-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Para Prevenir Insomnio", className="bg-info text-white"),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("Implementar técnicas de manejo del estrés"),
                                    html.Li("Mantener rutinas de sueño regulares"),
                                    html.Li("Aumentar gradualmente la actividad física")
                                ])
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Para Prevenir Apnea del Sueño", className="bg-warning text-white"),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("Mantener un peso saludable"),
                                    html.Li("Evitar el consumo de alcohol antes de dormir"),
                                    html.Li("Dormir de lado en lugar de boca arriba")
                                ])
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Recomendaciones Generales", className="bg-success text-white"),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("Mantener entorno de sueño adecuado (oscuro, silencioso, fresco)"),
                                    html.Li("Evitar pantallas antes de dormir"),
                                    html.Li("Limitar cafeína en las tardes")
                                ])
                            ])
                        ])
                    ], md=4)
                ]),
                
                html.H4("Aplicaciones Futuras", className="mb-3 mt-4"),
                html.Ul([
                    html.Li("Desarrollo de una aplicación de screening para identificar personas en riesgo de trastornos del sueño."),
                    html.Li("Implementación de programas de intervención personalizados basados en los factores de riesgo específicos."),
                    html.Li("Integración del modelo con datos de wearables para monitoreo continuo y detección temprana.")
                ])
            ], style=card_style)
        ], fluid=True)
    ]
)

# Layout principal reorganizado
app.layout = dbc.Container([
    navbar,
    dbc.Tabs([
        introduccion_tab,
        contexto_tab,
        problema_tab,
        objetivos_tab,
        marco_tab,
        metodologia_tab,
        resultados_tab,
        conclusiones_tab
    ], id='tabs', active_tab='tab-1', style={'marginTop': '20px'})
], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})

# =============================================
# EJECUCIÓN
# =============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=True, host="0.0.0.0", port=port)
