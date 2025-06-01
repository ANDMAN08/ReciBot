#----------------------------------------------------------------------------------------------------------------------------------------
# Autores: José Carlos Cordón, Gunther Franke, Angel Paiz, Manuel Pérez
# Carné: 25, 25, 25121, 23597
# Fecha: 03 de junio de 2025
# Descripción: App ReciBot
# Curso: Algoritmos y Programación Básica Sección: 90
#----------------------------------------------------------------------------------------------------------------------------------------




#----------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------Importamos todas las librerías que necesita el código para funcionar------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd                 # Manejo avanzado de datos tabulares (DataFrames) y persistencia en CSV
import streamlit as st             # Framework para crear la interfaz web interactiva de la aplicación
import os                         # Interacción con el sistema de archivos para gestión de archivos y directorios
from collections import defaultdict
import matplotlib.pyplot as plt   # Generación de gráficos y visualizaciones estadísticas para análisis de datos
import streamlit_survey as ss     # (Opcional) Soporte para encuestas y formularios avanzados dentro de Streamlit
from datetime import datetime, timedelta, date  # Gestión y manipulación precisa de fechas y tiempos
import numpy as np                # Operaciones numéricas, manejo de arreglos y soporte para cálculos estadísticos y gráficos


#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------Funciones para calculos estadísticos y almacenamiento de datos sobre la basura------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------


# -------------------------Función 1: obtener los datos que se usarán para el formulario-------------------------------------------------
def obtener_datos_usuario_formulario(nombre_usuario, fecha_registro, organico, plastico, papel, vidrio, metal, no_reciclable):
    """
    Crea una estructura de datos (lista con diccionario) que almacena la información básica
    ingresada por el usuario en el formulario, incluyendo nombre, fecha y cantidades de residuos por tipo.

    Parámetros:
    - nombre_usuario (str): Nombre o identificador del usuario.
    - fecha_registro (datetime): Fecha de registro de los datos.
    - organico, plastico, papel, vidrio, metal, no_reciclable (float): Cantidades (kg) de cada tipo de residuo.

    Retorna:
    - List[dict]: Lista con un diccionario que contiene los datos estructurados.
    """
    return [{
        "usuario": nombre_usuario,
        "fecha": fecha_registro.strftime("%Y-%m-%d"),
        "organico": organico,
        "plastico": plastico,
        "papel": papel,
        "vidrio": vidrio,
        "metal": metal,
        "no_reciclable": no_reciclable
    }]

# ----------------------Función 2: procesamiento de datos acumulados en diferentes periodos (semanal, mensual, anual)--------------------
def procesar_datos_basura(datos: list, kg_por_bolsa: float = 3.0, fecha_referencia: str = None) -> tuple:
    """
    Procesa la lista de registros de residuos y acumula las cantidades para periodos semanal, mensual y anual.
    Calcula además el número estimado de bolsas requeridas, dado un peso por bolsa.

    Parámetros:
    - datos (list): Lista de diccionarios con registros de residuos.
    - kg_por_bolsa (float): Peso estándar (kg) considerado para cada bolsa de basura.
    - fecha_referencia (str, opcional): Fecha en formato "YYYY-MM-DD" para definir el periodo de cálculo. 
      Si no se provee, se usa la fecha actual.

    Retorna:
    - tuple: Cuatro diccionarios con acumulados de bolsas, semanal, mensual y anual, respectivamente.
    """
    if fecha_referencia is None:
        fecha_ref = datetime.today()
    else:
        fecha_ref = datetime.strptime(fecha_referencia, "%Y-%m-%d")

    tipos = ["organico", "plastico", "papel", "vidrio", "metal", "no_reciclable"]

    semanal = {}
    mensual = {}
    anual = {}
    bolsas = {}

    for registro in datos:
        fecha_registro = datetime.strptime(registro["fecha"], "%Y-%m-%d")

        # Acumula valores anuales
        if fecha_registro.year == fecha_ref.year:
            for tipo in tipos:
                anual[tipo] += registro.get(tipo, 0.0)

        # Acumula valores mensuales
        if fecha_registro.year == fecha_ref.year and fecha_registro.month == fecha_ref.month:
            for tipo in tipos:
                mensual[tipo] += registro.get(tipo, 0.0)

        # Acumula valores semanales basados en la semana ISO
        if (fecha_registro.isocalendar()[0] == fecha_ref.isocalendar()[0] and
            fecha_registro.isocalendar()[1] == fecha_ref.isocalendar()[1]):
            for tipo in tipos:
                semanal[tipo] += registro.get(tipo, 0.0)

    # Calcula número de bolsas requerido por tipo basado en la suma semanal y kg por bolsa
    for tipo in tipos:
        bolsas[tipo] = round(semanal[tipo] / kg_por_bolsa, 2)
        semanal[tipo] = round(semanal[tipo], 2)
        mensual[tipo] = round(mensual[tipo], 2)
        anual[tipo] = round(anual[tipo], 2)

    return bolsas, dict(semanal), dict(mensual), dict(anual)

# ----------------------Función 3: almacenamiento de datos procesados en archivo CSV-----------------------------------------------
def guardar_datos_en_csv(datos: dict, bolsas: dict,
                         semanal: dict, mensual: dict, anual: dict,
                         ruta_csv: str = "datos_basura.csv") -> pd.DataFrame:
    """
    Guarda un nuevo registro consolidado en un archivo CSV. Si el archivo existe, añade la nueva fila; 
    si no, crea el archivo con los datos iniciales.

    Parámetros:
    - datos (dict): Diccionario con datos del usuario y fecha.
    - bolsas, semanal, mensual, anual (dict): Diccionarios con acumulados calculados de basura.
    - ruta_csv (str): Ruta o nombre del archivo CSV donde se guardan los datos.

    Retorna:
    - pd.DataFrame: DataFrame actualizado con todos los registros almacenados.
    """
    fila = {
        "Usuario": datos["usuario"],
        "Fecha": datos["fecha"],
        **{f"Bolsas_{k}": v for k, v in bolsas.items()},
        **{f"Semanal_{k}": v for k, v in semanal.items()},
        **{f"Mensual_{k}": v for k, v in mensual.items()},
        **{f"Anual_{k}": v for k, v in anual.items()}
    }

    if os.path.exists(ruta_csv):
        df_existente = pd.read_csv(ruta_csv)
        df_nuevo = pd.concat([df_existente, pd.DataFrame([fila])], ignore_index=True)
    else:
        df_nuevo = pd.DataFrame([fila])

    df_nuevo.to_csv(ruta_csv, index=False)
    return df_nuevo

# ----------------------Función 4: carga de datos desde archivo CSV-------------------------------------------------------------
def cargar_datos_csv(ruta_csv: str) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo CSV. Si no existe el archivo, retorna un DataFrame vacío.

    Parámetros:
    - ruta_csv (str): Ruta del archivo CSV a cargar.

    Retorna:
    - pd.DataFrame: DataFrame con los datos cargados o vacío si el archivo no existe.
    """
    if not pd.io.common.file_exists(ruta_csv):
        return pd.DataFrame()
    return pd.read_csv(ruta_csv)

# ----------------------Función 5: para filtrar datos por usuario y/o fecha----------------------------------------------------------------
def filtrar_datos(df: pd.DataFrame, usuario: str, fecha: str) -> pd.DataFrame:
    """
    Filtra un DataFrame según parámetros opcionales de usuario y fecha.

    Parámetros:
    - df (pd.DataFrame): DataFrame original.
    - usuario (str): Texto para filtrar por nombre de usuario (insensible a mayúsculas).
    - fecha (str): Fecha en formato "YYYY-MM-DD" para filtrar por fecha exacta.

    Retorna:
    - pd.DataFrame: DataFrame filtrado según criterios indicados.
    """
    if usuario:
        df = df[df["Usuario"].str.contains(usuario, case=False)]
    if fecha:
        df = df[df["Fecha"] == fecha]
    return df

# ----------------------Función 6: para obtener tabla filtrada (sin cambios)----------------------------------------------------------------
def obtener_tabla_filtrada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna el DataFrame tal cual, puede usarse para encapsular lógica adicional de filtrado o
    procesamiento previo si se requiere en el futuro.

    Parámetros:
    - df (pd.DataFrame): DataFrame a devolver.

    Retorna:
    - pd.DataFrame: DataFrame sin modificaciones.
    """
    return df

# ----------------------Función 7: para visualización gráfica usando matplotlib----------------------------------------------------------
def obtener_figura_bolsas(df: pd.DataFrame) -> plt.Figure:
    """
    Genera una gráfica de barras que muestra la suma total de bolsas usadas por tipo de residuo.

    Parámetros:
    - df (pd.DataFrame): DataFrame con datos de bolsas.

    Retorna:
    - matplotlib.figure.Figure: Figura con la gráfica de barras.
    """
    columnas_bolsas = [col for col in df.columns if col.startswith("Bolsas_")]
    suma_bolsas = df[columnas_bolsas].sum()

    fig, ax = plt.subplots()
    suma_bolsas.plot(kind='bar', ax=ax, color='darkgreen')
    ax.set_ylabel("Cantidad de bolsas")
    ax.set_xlabel("Tipo de residuo")
    ax.set_title("Total de bolsas usadas por tipo")

    return fig

# ----------------------Función 8: para visualización gráfica usando matplotlib----------------------------------------------------------
def obtener_figura_temporal(df: pd.DataFrame, periodo: str) -> plt.Figure | None:
    """
    Genera un gráfico circular (pie chart) que representa la distribución porcentual de residuos
    para un periodo específico: semanal, mensual o anual.

    Parámetros:
    - df (pd.DataFrame): DataFrame con datos acumulados.
    - periodo (str): Periodo para filtrar columnas (ejemplo: "Semanal", "Mensual", "Anual").

    Retorna:
    - matplotlib.figure.Figure o None: Figura con gráfico circular o None si no hay datos.
    """
    columnas = [col for col in df.columns if col.startswith(f"{periodo}_")]
    if not columnas:
        return None

    totales = df[columnas].sum()

    fig, ax = plt.subplots()
    totales.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title(f"Distribución de basura {periodo.lower()}")

    return fig

# ----------------------Función 9: para visualización gráfica usando matplotlib----------------------------------------------------------
def grafica_barras_agrupadas_por_usuario(df: pd.DataFrame) -> plt.Figure:
    """
    Genera un gráfico de barras agrupadas por usuario para comparar la cantidad de bolsas
    usadas según tipo de residuo.

    Parámetros:
    - df (pd.DataFrame): DataFrame con datos de bolsas y usuarios.

    Retorna:
    - matplotlib.figure.Figure: Figura con gráfico de barras agrupadas.
    """
    columnas_bolsas = [col for col in df.columns if col.startswith("Bolsas_")]
    
    datos = df[["Usuario"] + columnas_bolsas].copy()
    datos.rename(columns={col: col.replace("Bolsas_", "") for col in columnas_bolsas}, inplace=True)
    
    datos_agrupados = datos.groupby("Usuario").sum()
    
    tipos_basura = datos_agrupados.columns.tolist()
    usuarios = datos_agrupados.index.tolist()
    
    n_usuarios = len(usuarios)
    n_tipos = len(tipos_basura)
    ancho_barra = 0.8 / n_tipos
    
    x = np.arange(n_usuarios)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, tipo in enumerate(tipos_basura):
        ax.bar(x + i * ancho_barra, datos_agrupados[tipo], width=ancho_barra, label=tipo)
    
    ax.set_xticks(x + ancho_barra * (n_tipos - 1) / 2)
    ax.set_xticklabels(usuarios, rotation=45, ha="right")
    ax.set_ylabel("Cantidad de bolsas")
    ax.set_xlabel("Usuario")
    ax.set_title("Clasificación de basura por usuario y tipo (barras agrupadas)")
    ax.legend(title="Tipo de basura")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig


#---------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------Aqui empieza el código de la interfaz---------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

#-------------------------------------Ruta base del script para hallar el logo y el page icon-------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "Logo.png")
ICONO_PATH = os.path.join(BASE_DIR, "app.ico")

#-----------------------------------------------Configuración de la página--------------------------------------------------

st.set_page_config(page_title="ReciBot", page_icon=ICONO_PATH, layout="centered")
# Footer or info section
st.markdown("---")  # horizontal separator line

last_update = datetime(2025, 5, 31, 10, 0)  # example last update time



#------------------------------------------------Mostrar logo si existe-----------------------------------------------------

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=1100)
else:
    st.warning("⚠️ No se encontró el archivo 'Logo.png'. Asegúrate de que esté en la misma carpeta que este script.")

#------------------------------------------------Variables contadoras------------------------------------------------------
organico = 0
reciclable = 0
no_reciclable = 0
papel = 0
especial = 0

basura = {
    "Orgánico": organico,
    "Reciclable": reciclable,
    "No Reciclable": no_reciclable,
    "Papel": papel,
    "Especial(Punto limpio)": especial
}

#-------------------------------------------------------Preguntas----------------------------------------------------------

Q1 = "♻️ ¿El objeto es un residuo sólido o líquido?"
Q2 = "🍎 ¿El objeto es parte de un alimento o proviene de algo natural (fruta, verdura, carne, huevo, pan, flores, hojas)?"
Q3 = "🔥 ¿Está cocinado o tiene grasa?"
Q4 = "📄 ¿Está hecho principalmente de papel o cartón?"
Q5 = "💧 ¿Tiene restos de comida, está mojado o tiene grasa?"
Q6 = "📃 ¿Es papel blanco, hojas impresas, libretas o cajas de cartón?"
Q7 = "🧴 ¿El residuo es de vidrio, plástico o metal?"
Q8 = "✨ ¿Está limpio y sin restos de comida o líquido?"
Q9 = "🔍 Tipo de material:"
Q10 = "🚼 ¿Se trata de objetos higiénicos o personales (pañales, toallas sanitarias, cotonetes, colillas)?"
Q11 = "👕 ¿Son textiles, ropa vieja o zapatos?"
Q12 = "📱 ¿Es un aparato electrónico, pila o foco?"
Q13 = "💊 ¿Es un medicamento, jeringa, o químico (pintura, aceite, etc.)?"
error = "❌ No es posible encontrar tu basura, deséchala en el centro de recuperación más cercano usando un contenedor sellado"

#--------------------------------------------------------------Opciones---------------------------------------------------------------------------

opciones = ["Bienvenida", "Formulario de clasificación", "Preguntas Frecuentes", "Ingresar basura para estadística", "mostrar basura"]
seccion = st.sidebar.selectbox("MENU", opciones)


#-----------------------------------------------------------Bienvenida----------------------------------------------------------------

if seccion == "Bienvenida":
    st.markdown("---")
    st.subheader("")
    st.subheader("¡Ya puedes empezar a utilizar nuestra interfaz!")
    st.markdown(
        """
        <p style='font-size:18px; color: #2E86C1;'>
        Puedes empezar a usar la aplicación seleccionando alguna de las opciones en la barra lateral. 
        <br>¡Recuerda responder con sinceridad para obtener resultados precisos! 😊
        </p>
        """,
        unsafe_allow_html=True
    )
#----------------------------------------------------------Formulario----------------------------------------------------------------
elif seccion == "Formulario de clasificación":
    st.markdown("## 🗑️ Formulario de Clasificación de Residuos")
    st.markdown(
        """
        <p style='font-size:18px; color:#117A65;'>
        Por favor, responde las siguientes preguntas para clasificar correctamente tu basura. 
        </p>
        <p style='font-size:16px; color:#1F618D;'>
        Para todas las opciones, escribe el valor numérico correspondiente a la opción que deseas seleccionar. 🔢
        </p>
        """,
        unsafe_allow_html=True
    )

    survey = ss.StreamlitSurvey("Survey 1")

    Respuesta1 = survey.radio(f"🟢 {Q1}", options=["Sólido", "Líquido"])
    if Respuesta1 == "Sólido":
        Respuesta2 = survey.radio(f"🍎 {Q2}", options=["Sí", "No"])
        if Respuesta2 == "Sí":
            Respuesta3 = survey.radio(f"🍳 {Q3}", options=["Sí", "No"])
            if Respuesta3 == "Sí":
                st.markdown("### ✅ Deséchalo en: **Contenedor Orgánico** 🌱")
                organico += 1
            elif Respuesta3 == "No":
                SubRes3 = survey.radio("🌰 ¿Es cáscara, semilla, hueso o vegetal?", options=["Sí", "No"])
                if SubRes3 == "Sí":
                    st.markdown("### ✅ Deséchalo en: **Contenedor Orgánico** 🌿")
                    organico += 1
                elif SubRes3 == "No":
                    st.markdown(f"### ❌ {error} ⚠️")
        elif Respuesta2 == "No":
            Respuesta4 = survey.radio(f"📄 {Q4}", options=["Sí", "No"])
            if Respuesta4 == "Sí":
                Respuesta5 = survey.radio(f"🍔 {Q5}", options=["Sí", "No"])
                if Respuesta5 == "Sí":
                    st.markdown("### 🚫 Deséchalo en: **Contenedor No Reciclable** 🗑️")
                    no_reciclable += 1
                elif Respuesta5 == "No":
                    Respuesta6 = survey.radio(f"📚 {Q6}", options=["Sí", "No"])
                    if Respuesta6 == "Sí":
                        st.markdown("### ♻️ Deséchalo en: **Contenedor de Papel** 📦")
                        papel += 1
                    elif Respuesta6 == "No":
                        SubRes6 = survey.radio("🧻 ¿Es papel encerado, plastificado o papel higiénico?", options=["Sí", "No"])
                        if SubRes6 == "Sí":
                            st.markdown("### 🚫 Deséchalo en: **Contenedor No Reciclable** 🗑️")
                            no_reciclable += 1
                        elif SubRes6 == "No":
                            st.markdown(f"### ❌ {error} ⚠️")
            elif Respuesta4 == "No":
                Respuesta7 = survey.radio(f"🧴 {Q7}", options=["Sí", "No"])
                if Respuesta7 == "Sí":
                    Respuesta8 = survey.radio(f"🧼 {Q8}", options=["Sí", "No"])
                    if Respuesta8 == "Sí":
                        Respuesta9 = survey.radio(f"🔍 {Q9}", options=[
                            "Plástico PET (botellas, envases)",
                            "Vidrio (botellas, frascos sin tapa)",
                            "Latas de aluminio",
                            "Plástico duro, mezclado o bolsas sucias"
                        ])
                        if Respuesta9 in ["Plástico PET (botellas, envases)", "Vidrio (botellas, frascos sin tapa)", "Latas de aluminio"]:
                            st.markdown("### ♻️ Deséchalo en: **Contenedor Reciclable** 🔄")
                            reciclable += 1
                        elif Respuesta9 == "Plástico duro, mezclado o bolsas sucias":
                            st.markdown("### 🚫 Deséchalo en: **Contenedor No Reciclable** 🗑️")
                            no_reciclable += 1
                    elif Respuesta8 == "No":
                        st.markdown("### 🚫 Deséchalo en: **Contenedor No Reciclable** (A menos que se lave antes) 🧼")
                        no_reciclable += 1
                elif Respuesta7 == "No":
                    Respuesta10 = survey.radio(f"🧻 {Q10}", options=["Sí", "No"])
                    if Respuesta10 == "Sí":
                        st.markdown("### 🚫 Deséchalo en: **Contenedor No Reciclable** 🗑️")
                        no_reciclable += 1
                    elif Respuesta10 == "No":
                        Respuesta11 = survey.radio(f"👕 {Q11}", options=["Sí", "No"])
                        if Respuesta11 == "Sí":
                            st.markdown("### 🚫 Deséchalo en: **Contenedor No Reciclable** (Si es posible, llévalo a un punto de reciclaje textil o, si están rotos o sucios) 🧺")
                            no_reciclable += 1
                        elif Respuesta11 == "No":
                            Respuesta12 = survey.radio(f"🔋 {Q12}", options=["Sí", "No"])
                            if Respuesta12 == "Sí":
                                st.markdown("### ⚠️ No se debe tirar en contenedores comunes. Llévalo a un punto limpio o reciclaje electrónico 🔌")
                                especial += 1
                            elif Respuesta12 == "No":
                                Respuesta13 = survey.radio(f"💊 {Q13}", options=["Sí", "No"])
                                if Respuesta13 == "Sí":
                                    st.markdown("### ⚠️ Punto limpio o farmacia autorizada. Nunca en contenedor común. 🏥")
                                    especial += 1
                                elif Respuesta13 == "No":
                                    st.markdown(f"### ❌ {error} ⚠️")
    elif Respuesta1 == "Líquido":
        st.markdown("### ⚠️ NO debe desecharse en contenedor común. Llévalo a un punto limpio especializado. 🚱")
        especial += 1

#-----------------------------------------------------------Preguntas Frecuentes----------------------------------------------------------------

elif seccion == "Preguntas Frecuentes":
    st.title("Preguntas Frecuentes - Clasificación de Basura (Guatemala 2025)")
    st.markdown("Selecciona una pregunta para ver su respuesta:")

    # Diccionario de preguntas y respuestas
    preguntas_respuestas = {
        "1. ¿Qué tipos de residuos se deben clasificar en Guatemala?":
            "Orgánico, Reciclable, No Reciclable, Papel y Especial (Punto limpio).",
        "2. ¿Qué va en el contenedor orgánico?":
            "Restos de alimentos naturales como frutas, verduras, cáscaras, semillas, huesos, pan, etc.",
        "3. ¿Qué residuos son reciclables?":
            "Plástico PET, vidrio limpio, latas de aluminio y papel o cartón limpio.",
        "4. ¿El papel mojado se puede reciclar?":
            "No, debe ir al contenedor No Reciclable.",
        "5. ¿Dónde va el papel blanco o impreso limpio?":
            "Contenedor de Papel.",
        "6. ¿Qué hago con pañales y toallas sanitarias?":
            "Van en el contenedor No Reciclable.",
        "7. ¿Y con ropa vieja o textiles?":
            "Preferiblemente llevarlos a reciclaje textil. Si están sucios o rotos, van en el No Reciclable.",
        "8. ¿Dónde van electrónicos, pilas y focos?":
            "Punto limpio o reciclaje electrónico autorizado.",
        "9. ¿Cómo desechar medicamentos vencidos o jeringas?":
            "Llevar a farmacias autorizadas o puntos limpios.",
        "10. ¿Qué hacer con pintura o aceite usado?":
            "Desechar en puntos limpios. Nunca en el drenaje o contenedor común.",
        "11. ¿Las bolsas plásticas se reciclan?":
            "Solo si están limpias y secas.",
        "12. ¿Qué hago con empaques de comida rápida?":
            "Si tienen grasa o residuos, van al No Reciclable.",
        "13. ¿Dónde van los empaques tipo tetrapack?":
            "Limpios y secos pueden reciclarse. Sucios, al No Reciclable.",
        "14. ¿Las botellas de plástico son siempre reciclables?":
            "Sí, si están limpias, secas y vacías.",
        "15. ¿Qué pasa si mezclo residuos?":
            "Contaminas materiales reciclables y dificultas su aprovechamiento.",
        "16. ¿Debo enjuagar los reciclables?":
            "Sí, siempre deben estar limpios y secos.",
        "17. ¿Qué es un punto limpio?":
            "Centro especializado para residuos peligrosos o electrónicos.",
        "18. ¿El papel higiénico es reciclable?":
            "No. Va en el No Reciclable.",
        "19. ¿Dónde reporto un punto limpio dañado?":
            "Municipalidad o Ministerio de Ambiente.",
        "20. ¿Las empresas deben clasificar basura?":
            "Sí, es obligatorio según la normativa 2025.",
        "21. ¿Dónde van los utensilios de madera como palillos o paletas?":
            "Si no están sucios, pueden ir al contenedor Orgánico. Si tienen residuos, al No Reciclable.",
        "22. ¿Dónde tiro los cepillos de dientes?":
            "Contenedor No Reciclable o reciclaje especializado si es de bambú o reciclable.",
        "23. ¿Las servilletas usadas se reciclan?":
            "No. Van en el contenedor No Reciclable.",
        "24. ¿Qué hago con papel aluminio?":
            "Limpio, puede reciclarse. Sucio, No Reciclable.",
        "25. ¿Dónde van los vasos de cartón encerado?":
            "Al contenedor No Reciclable.",
        "26. ¿Puedo reciclar CDs o DVDs?":
            "No en el reciclaje tradicional. Pueden llevarse a puntos limpios si se aceptan.",
        "27. ¿Qué hago con las cajas de pizza?":
            "Partes limpias van al contenedor de Papel. Las grasosas, al No Reciclable.",
        "28. ¿Los juguetes rotos se reciclan?":
            "En general no. Van al No Reciclable, salvo excepciones si están limpios y son de plástico.",
        "29. ¿Cómo descartar esponjas de cocina?":
            "Van en el contenedor No Reciclable.",
        "30. ¿Qué pasa si tiro residuos peligrosos en la basura común?":
            "Contaminas el medio ambiente y puedes causar accidentes.",
        "31. ¿Los cubiertos plásticos son reciclables?":
            "Solo si están limpios y si el centro acepta ese tipo de plástico.",
        "32. ¿Dónde van las botellas de vidrio rotas?":
            "En el reciclaje de vidrio, bien protegidas y limpias.",
        "33. ¿Dónde tiro el aceite de cocina usado?":
            "Debe almacenarse en botellas cerradas y llevarse a un punto limpio.",
        "34. ¿Cómo clasificar residuos de jardín?":
            "Hojas, ramas pequeñas y flores van en Orgánico.",
        "35. ¿Dónde tiro materiales de construcción como cemento o yeso?":
            "Deben gestionarse como residuos especiales en puntos autorizados.",
        "36. ¿Los sorbetes (popotes) son reciclables?":
            "No, en general van en el contenedor No Reciclable.",
        "37. ¿Qué hacer con termos, floreros o cerámica rota?":
            "Van en el contenedor No Reciclable o en desechos especiales.",
        "38. ¿Cómo se clasifican las tapas plásticas?":
            "Reciclables si están limpias. Se recomienda separarlas.",
        "39. ¿Las cajas de huevo se reciclan?":
            "Sí, si son de cartón seco. Las de espuma van al No Reciclable.",
        "40. ¿Qué hago con cables o cargadores dañados?":
            "Llevar a reciclaje electrónico o punto limpio.",
        "41. ¿Los plumones y marcadores se reciclan?":
            "No. Van en el contenedor No Reciclable.",
        "42. ¿Las botellas con etiquetas se pueden reciclar?":
            "Sí. Se recomienda quitar la etiqueta si es posible.",
        "43. ¿Cómo clasifico los envoltorios de golosinas?":
            "Van en el contenedor No Reciclable.",
        "44. ¿Se reciclan los envases de yogurt?":
            "Sí, si están completamente limpios y secos.",
        "45. ¿Dónde tiro el cartón de huevo mojado?":
            "Va al contenedor No Reciclable.",
        "46. ¿Qué hago con los botes de desodorante en aerosol?":
            "Punto limpio. Son residuos presurizados y peligrosos.",
        "47. ¿Puedo reciclar frascos de vidrio con tapa?":
            "Sí, pero es mejor separar la tapa (metal/plástico).",
        "48. ¿Qué es reciclaje mixto?":
            "Es cuando varios materiales reciclables se recogen juntos para su posterior separación.",
        "49. ¿Puedo reciclar botellas con tapas?":
            "Sí. Si el centro las acepta así, incluso mejor.",
        "50. ¿Dónde encuentro los centros de reciclaje en Guatemala?":
            "Consulta en el portal del Ministerio de Ambiente o municipalidades locales."
    }

    # Clave persistente: número de preguntas mostradas
    if "preguntas_mostradas" not in st.session_state:
        st.session_state.preguntas_mostradas = 10  # mostrar primeras 10

    # Lista de todas las preguntas
    lista_preguntas = list(preguntas_respuestas.items())

    # Mostrar preguntas actuales
    for i in range(st.session_state.preguntas_mostradas):
        pregunta, respuesta = lista_preguntas[i]
        with st.expander(pregunta):
            st.write(respuesta)

    # Botón para mostrar más
    if st.session_state.preguntas_mostradas < len(lista_preguntas):
        if st.button("🔽 Ver más preguntas"):
            st.session_state.preguntas_mostradas += 5

# --------------------------------------------------Ingreso de datos de basura---------------------------------------------------

elif seccion == "Ingresar basura para estadística":
    st.subheader("Rellene los datos siguientes:")

    # Entrada de texto para capturar el nombre del usuario
    nombre_usuario = st.text_input("Nombre del usuario")

    # Entrada de fecha para registrar el día de la recolección o registro
    fecha_registro = st.date_input("Fecha del registro", value=date.today())

    st.markdown("### Ingrese la cantidad de basura generada (en kilogramos):")

    # Entradas numéricas para las cantidades (en kg) de distintos tipos de residuos
    organico = st.number_input("Orgánica", min_value=0.0, step=0.1)
    plastico = st.number_input("Plástico", min_value=0.0, step=0.1)
    papel = st.number_input("Papel y cartón", min_value=0.0, step=0.1)
    vidrio = st.number_input("Vidrio", min_value=0.0, step=0.1)
    metal = st.number_input("Metal", min_value=0.0, step=0.1)
    no_reciclable = st.number_input("No reciclable", min_value=0.0, step=0.1)

    # Botón para guardar la información ingresada por el usuario
    if st.button("Guardar información"):

        # Validación para asegurarse que el nombre de usuario no esté vacío
        if nombre_usuario.strip() == "":
            st.warning("⚠️ Debe ingresar un nombre.")
        else:
            # Se obtienen los datos en formato de diccionario para procesar y almacenar
            datos_usuario = obtener_datos_usuario_formulario(
                nombre_usuario, fecha_registro,
                organico, plastico, papel,
                vidrio, metal, no_reciclable
            )

            # Procesar los datos ingresados para calcular acumulados por bolsas, semanal, mensual y anual
            bolsas, semanal, mensual, anual = procesar_datos_basura(datos_usuario)

            # Guardar los datos procesados en un archivo CSV y obtener DataFrame actualizado
            df_resultado = guardar_datos_en_csv(
                datos_usuario[0], bolsas, semanal, mensual, anual
            )

            # Confirmación visual de que los datos fueron guardados correctamente
            st.success("✅ Datos guardados correctamente.")

            # Mostrar el último registro guardado en forma tabular para revisión inmediata
            st.dataframe(df_resultado.tail(1))

# --------------------------------------------------Análisis estadístico--------------------------------------------------------

elif seccion == "mostrar basura":
    st.header("📊 Visualización de Datos Recopilados")

    # Definir ruta del archivo CSV donde se almacenan los datos
    ruta_csv = "datos_basura.csv"

    # Cargar los datos existentes en un DataFrame
    df = cargar_datos_csv(ruta_csv)

    # Validar si existen datos para mostrar
    if df.empty:
        st.warning("⚠️ Aún no hay datos almacenados.")
    else:
        # Obtener lista ordenada de usuarios y fechas para filtros
        usuarios = sorted(df["Usuario"].unique())
        fechas = sorted(df["Fecha"].unique())

        # Selector para filtrar por usuario
        usuario_filtro = st.selectbox("🔍 Buscar por usuario", options=[""] + usuarios)

        # Selector para filtrar por fecha
        fecha_filtro = st.selectbox("📅 Buscar por fecha", options=[""] + fechas)

        # Aplicar filtros seleccionados para obtener subconjunto de datos
        df_filtrado = filtrar_datos(df, usuario_filtro, fecha_filtro)

        # Validar si el filtro generó datos para mostrar
        if df_filtrado.empty:
            st.info("No se encontraron registros con esos filtros.")
        else:
            # Mostrar tabla con datos filtrados
            tabla = obtener_tabla_filtrada(df_filtrado)
            st.markdown("### 📋 Datos filtrados:")
            st.dataframe(tabla)

            # Generar y mostrar gráfica de barras con cantidad de bolsas por tipo de basura
            figura_bolsas = obtener_figura_bolsas(df_filtrado)
            st.markdown("### 🛍️ Cantidad de bolsas por tipo de basura")
            st.pyplot(figura_bolsas)

            # Mostrar gráficos circulares para distribución de basura en períodos: semanal, mensual y anual
            for periodo in ["Semanal", "Mensual", "Anual"]:
                figura_temporal = obtener_figura_temporal(df_filtrado, periodo)
                if figura_temporal:
                    st.markdown(f"### 📈 Distribución de basura {periodo.lower()}")
                    st.pyplot(figura_temporal)

        # Gráfica comparativa de barras agrupadas por usuario y tipo de basura
        fig = grafica_barras_agrupadas_por_usuario(df)
        st.markdown("### 👥 Comparación entre usuarios")
        st.pyplot(fig)

# Pie de página con información de autoría y fecha de actualización
st.markdown("""
<div style="font-size: 0.8em; color: gray;">
    Developed by: UVG Students <br>
    Last updated: {update}<br>
    © 2025 All rights reserved.
</div>
""".format(update=last_update.strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
