import pandas as pd
import csv
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

x = ([1, 2, 3, 4, 5,6,7,8,9,10])
y = ([1,1.2,1.3	,1.2,1.4,1.2, 1.21, 1.45,1.183,1.411])


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
    print(f"Para el grado {i} la prediccion es {p} ")

    x1 = np.linspace(1, mes + 1, 10)
y1 = fx(x1, coef)
plt.figure(figsize=[20, 10])
plt.title("Cantidad por periodo : " + str(i))

plt.scatter(x, y, s=120, c='blueviolet')
plt.plot(x1, y1, "--", linewidth=3, color="orange")
plt.scatter(mes, p, s=20, c="red")
plt.yticks(range(10, 50, 20))
plt.grid("on")
ax = plt.gca()
ax.set_xlabel("$Mes$")
ax.set_ylabel("$cantidad$")
plt.show()

mes = 13
grado = np.arange(0, 10 + 1, 1)
aproxi = np.array([])
y_pred_vec = np.array([])
for i in grado:
    coef = np.polyfit(x, y, i)
    p = np.polyval(coef, mes)
    aproxi = np.append(aproxi, p)

    # y_pred_vec= np.array([])

plt.title("grado del polinomio vs la presion mostrada")
plt.plot(grado, aproxi, "--", linewidth=3, color='red')
plt.grid("on")
plt.show()