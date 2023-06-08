from itertools import product

# Precios de venta de cada licencia
precios = [3000, 20000, 50000, 100000]

# Cantidad total de licencias
total_licencias = 100

max_ganancia_3000 = 0
mejor_combinacion_3000 = None

for cantidades in product(range(total_licencias // 2 + 1), repeat=4):
    if sum(cantidades) == total_licencias and cantidades[0] > 0 and cantidades[0] <= total_licencias // 2:
        ganancia = sum(precios[i] * cantidades[i] for i in range(4))
        if ganancia > max_ganancia_3000:
            max_ganancia_3000 = ganancia
            mejor_combinacion_3000 = cantidades

print("Caso 1: Vender más licencias de 3000")
print("Ganancia máxima:", max_ganancia_3000)
print("Mejor combinación:", mejor_combinacion_3000)
from itertools import product

# Precios de venta de cada licencia
precios = [3000, 20000, 50000, 100000]

# Cantidad total de licencias
total_licencias = 100

max_ganancia_20000 = 0
mejor_combinacion_20000 = None

for cantidades in product(range(total_licencias // 2 + 1), repeat=4):
    if sum(cantidades) == total_licencias and cantidades[1] > 0 and cantidades[1] <= total_licencias // 2:
        ganancia = sum(precios[i] * cantidades[i] for i in range(4))
        if ganancia > max_ganancia_20000:
            max_ganancia_20000 = ganancia
            mejor_combinacion_20000 = cantidades

print("Caso 2: Vender más licencias de 20000")
print("Ganancia máxima:", max_ganancia_20000)
print("Mejor combinación:", mejor_combinacion_20000)
from itertools import product

# Precios de venta de cada licencia
precios = [3000, 20000, 50000, 100000]

# Cantidad total de licencias
total_licencias = 100

max_ganancia_50000 = 0
mejor_combinacion_50000 = None

for cantidades in product(range(total_licencias // 2 + 1), repeat=4):
    if sum(cantidades) == total_licencias and cantidades[2] > 0 and cantidades[2] <= total_licencias // 2:
        ganancia = sum(precios[i] * cantidades[i] for i in range(4))
        if ganancia > max_ganancia_50000:
            max_ganancia_50000 = ganancia
            mejor_combinacion_50000 = cantidades

print("Caso 3: Vender más licencias de 50000")
print("Ganancia máxima:", max_ganancia_50000)
print("Mejor combinación:", mejor_combinacion_50000)
from itertools import product

# Precios de venta de cada licencia
precios = [3000, 20000, 50000, 100000]

# Cantidad total de licencias
total_licencias = 100

max_ganancia_100000 = 0
mejor_combinacion_100000 = None

for cantidades in product(range(total_licencias // 2 + 1), repeat=4):
    if sum(cantidades) == total_licencias and cantidades[3] > 0 and cantidades[3] <= total_licencias // 2:
        ganancia = sum(precios[i] * cantidades[i] for i in range(4))
        if ganancia > max_ganancia_100000:
            max_ganancia_100000 = ganancia
            mejor_combinacion_100000 = cantidades

print("Caso 4: Vender más licencias de 100000")
print("Ganancia máxima:", max_ganancia_100000)
print("Mejor combinación:", mejor_combinacion_100000)
