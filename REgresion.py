import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([33,64,25,39])


def fx(x1, coef):
    fx = 0
    n = len(coef) - 1
    for p in coef:
        fx = fx + p * x1 ** 2
        n = n - 1
    return fx


mes = 13
for i in range(0, 10):
    coef = np.polyfit(x, y, i)
    p = np.polyval(coef, mes)
    print(f"Para el grado {i}, la predicción es {p}")

    x1 = np.linspace(1, mes + 1, 10)
    y1 = fx(x1, coef)
    plt.figure(figsize=[20, 10])
    plt.title("Cantidad por periodo: " + str(i))

    plt.scatter(x, y, s=120, c='blueviolet')
    plt.plot(x1, y1, "--", linewidth=3, color="orange")
    plt.scatter(mes, p, s=20, c="red")
    plt.yticks(range(10, 50, 20))
    plt.grid(True)
    ax = plt.gca()
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad")
    plt.show()

mes = np.arange(5, 13)  # Próximos 8 periodos
grado = np.arange(2, 5)  # Grados 2 al 4
for i in grado:
    coef = np.polyfit(x, y, i)
    p = np.polyval(coef, mes)
    print(f"Grado {i}: {p}")

plt.title("Grado del polinomio vs Presión mostrada")
plt.plot(grado, aproxi, "--", linewidth=3, color='red')
plt.grid(True)
plt.show()
