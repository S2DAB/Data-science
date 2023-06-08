import pandas as pd

# Lee el archivo Excel
df = pd.read_excel('Datos_dolar.xlsx')

# Nombre de la tabla en la base de datos
nombre_tabla = 'nombre_tabla'

# Genera las sentencias SQL para insertar los datos
sql = []
for index, row in df.iterrows():
    values = "', '".join(str(value) for value in row.values)
    insert_statement = f"INSERT INTO {nombre_tabla} VALUES ('{values}')"
    sql.append(insert_statement)

# Guarda las sentencias SQL en un archivo
with open('archivo.sql', 'w') as file:
    file.write('\n'.join(sql))
