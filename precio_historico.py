import numpy as np
import matplotlib.pyplot as plt

x = [2018, 2019, 2020, 2021, 2022]
y = [1.4,1.8,1.8,2,2.1]

def fx(x1, coef):
    fx = 0
    n = len(coef) - 1
    for p in coef:
        fx = fx + p * x1 ** n
        n = n - 1
    return fx

year = 2022
for i in range(0, 10):
    coef = np.polyfit(x, y, i)
    p = np.polyval(coef, year)
    print(f"Para el grado {i}, la predicción es {p}")

years_pred = None
while years_pred is None:
    try:
        years_pred = int(input("Ingresa el número de años a predecir: "))
    except ValueError:
        print("Entrada inválida. Introduce un número entero.")

grado_pred = None
while grado_pred is None:
    try:
        grado_pred = int(input("Ingresa el grado del polinomio para la predicción: "))
    except ValueError:
        print("Entrada inválida. Introduce un número entero.")

coef_pred = np.polyfit(x, y, grado_pred)
p_pred = np.polyval(coef_pred, year + years_pred)
print(f"Para el grado {grado_pred}, la predicción para el año {year + years_pred} es {p_pred}")

years = np.arange(year, year + years_pred + 1)
predicciones = np.polyval(coef_pred, years)

print("Predicciones para los años:")
for y, prediccion in zip(years, predicciones):
    print(f"Año {y}: {prediccion}")

plt.figure(figsize=[20, 10])
plt.title(f"Predicción para años {year} a {year + years_pred}")
plt.scatter(x, y, s=120, c='blueviolet', label="Datos históricos")
plt.plot(years, predicciones, "--", linewidth=3, color="orange", label="Predicciones")
plt.scatter(year + years_pred, p_pred, s=20, c="red", label="Predicción para el futuro")
plt.yticks(np.arange(1, 2, 0.1))
plt.xticks(np.arange(year, year + years_pred + 1))
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.set_xlabel("Año")
ax.set_ylabel("Cantidad")
plt.show()
