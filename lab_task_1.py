import math
import matplotlib.pyplot as plt
import numpy as np


# Defining a function f(x, y)
def f(sumval1, sumval2):
	return (x[i] + sumval1) * math.exp(-((x[i] + sumval1) ** 2)) - 2 * (x[i] + sumval1) * (y2[i] + sumval2)


# Creating list of values from 0 to 1 with constant relative distance 0,05
h = 0.05
split_num = int(1 / h) + 1
x = np.linspace(0, 1, split_num)

# Creating Euler, Runge-Kutta and analytic lists of values
# Defining their initial conditions
y1 = np.zeros(len(x))
y1[0] = 1

y2 = np.zeros(len(x))
y2[0] = 1

y3 = np.zeros(len(x))
y3[0] = 1

# Filling the lists with values
for i in range(len(x) - 1):
	# Euler
	y1[i + 1] = y1[i] + h * (x[i] * math.exp(-(x[i] ** 2)) - 2 * x[i] * y1[i])

	# Runge-Kutta 4 orders
	k1 = h * f(0, 0)
	k2 = h * f(h / 2, k1 / 2)
	k3 = h * f(h / 2, k2 / 2)
	k4 = h * f(h, k3)
	y2[i + 1] = y2[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

	# Analytics
	y3[i + 1] = (x[i + 1] ** 2 + 2) / (2 * math.exp(x[i + 1] ** 2))

# Printing all y lists in format: y1 y2 y3 with precision up to 4 digits
print('{:<10} {:<10} {:<10} {:<10}'.format('x', 'Euler', 'R-K', 'Analytic'))
for i in range(len(x)):
	x_i = "%.2f" % x[i]
	y_euler = "%.4f" % y1[i]
	y_rk = y2[i]
	y_analytic = y3[i]
	print('{:<10} {:<10} {:<10} {:<10}'.format(x_i, y_euler, y_rk, y_analytic))

print('\n', '\n')

# Printing each y(b) and each solution errors
print(f'''Analytic:
y(b) = {"%.4f" % y3[-1]},

Euler:
y(b) = {"%.4f" % y1[-1]},
Euler error = {abs(float(y1[-1]) - float(y3[-1]))},

R-K:
y(b) = {"%.4f" % y2[-1]},
R-K error = {abs(float(y2[-1]) - float(y3[-1]))}.''')

# Plotting
plt.plot(x, y1, label="Euler")
plt.plot(x, y2, label="Runge-Kutta 4 orders", linestyle='--')
plt.plot(x, y3, label="Analytics", linestyle=':')
plt.legend()
plt.show()
