import pandas as pd
import arcpy

# Ruta al archivo Excel
excel_file = "MApaaYDatosExcel.xlsx"

# Leer el archivo Excel utilizando pandas
df = pd.read_excel(excel_file)

# Convertir el DataFrame de pandas a un arreglo de NumPy
data = df.values

# Ruta al archivo de geodatabase en ArcGIS Pro
geodatabase = "ruta_a_la_geodatabase.gdb"

# Nombre de la tabla en la geodatabase
tabla = "nombre_de_la_tabla"

# Crear una tabla en la geodatabase
arcpy.CreateTable_management(geodatabase, tabla)

# Obtener la lista de nombres de columnas del archivo Excel
nombres_columnas = df.columns.tolist()

# Agregar campos a la tabla en la geodatabase
for nombre_columna in nombres_columnas:
    arcpy.AddField_management(tabla, nombre_columna, "TEXT")

# Abrir un cursor de inserción para la tabla
with arcpy.da.InsertCursor(tabla, nombres_columnas) as cursor:
    # Insertar filas en la tabla
    for fila in data:
        cursor.insertRow(fila)

# Mostrar la tabla en ArcGIS Pro
arcpy.management.MakeTableView(tabla, "tabla_view")
arcpy.management.AddTableView("ruta_al_mapa", "tabla_view")
