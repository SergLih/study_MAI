# Фамилия: Лихарев С.С.
# Группа:  M80-307Б
# Программа аппроксимации до цилиндра

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

#выбор цвета
def choose_color(color, view):
	if color == 1:
	    cl = (1, 0, 0, view)
	if color == 2:
		cl = (1, 1, 1, view)
	if color == 3:
		cl = (1, 1, 0, view)
	if color == 4:
		cl = (0, 0, 0, view)
	return cl

#стороны призмы
def make_sides(p):
	sides = [[p[0],p[1],p[2],p[3],p[4],p[5]],
			 [p[6],p[7],p[8],p[9],p[10],p[11]],
			 [p[0],p[1],p[7],p[6]],
			 [p[1],p[2],p[8],p[7]],
			 [p[2],p[3],p[9],p[8]],
			 [p[3],p[4],p[10],p[9]],
			 [p[4],p[5],p[11],p[10]],
			 [p[5],p[0],p[6],p[11]]]
	return sides

#аппроксимация до цилиндра
def approximate(list_of_points, n):

	h = list_of_points[0][0][2]
	R = list_of_points[0][2][0]

	x_1 = np.array([-0.5 * R, 0.])
	x_2 = np.array([0.5 * R, 0.])
	x_3 = np.array([R, 0.])
	for i in range(1, n + 1):# просчитываем координаты новых призм.
		x_1[0] += R / n
		x_1[1] = sqrt(R ** 2 + 0.00000000000001 - x_1[0] ** 2)# const нужна, т.к. при округлении в корне может возникнуть отрицательное число

		x_2[0] += 0.5 * R / n
		x_2[1] = sqrt(R ** 2 + 0.00000000000001 - x_2[0] ** 2)

		x_3[0] -= 0.5 * R / n
		x_3[1] = -1 * sqrt(R ** 2 + 0.00000000000001 - x_3[0] ** 2)

		list_of_points += [[np.hstack((x_1, [-h])),
						   np.hstack((x_2, [-h])),# сшиваем две матрицы подходящих размерностей
						   np.hstack((x_3, [-h])),
						   np.hstack((x_1 * -1, [-h])),
						   np.hstack((x_2 * -1, [-h])),
						   np.hstack((x_3 * -1, [-h])),

						   np.hstack((x_1, [h])),
						   np.hstack((x_2, [h])),
						   np.hstack((x_3, [h])),
						   np.hstack((x_1 * -1, [h])),
						   np.hstack((x_2 * -1, [h])),
						   np.hstack((x_3 * -1, [h]))]]
	return list_of_points

#координаты
def plot_prism(R=1., height=2., alpha=20, color=3, view=0.2):
	points = np.array([[-0.5, sqrt(3) / 2, -1.],
					   [0.5, sqrt(3) / 2, -1.],
					   [1., 0., -1.],
					   [0.5, -sqrt(3) / 2, -1.],
					   [-0.5, -sqrt(3) / 2, -1.],
					   [-1., 0., -1.],
					   
					   [-0.5, sqrt(3) / 2, 1.],
					   [0.5, sqrt(3) / 2, 1.],
					   [1., 0., 1.],
					   [0.5, -sqrt(3) / 2, 1.],
					   [-0.5, -sqrt(3) / 2, 1.],
					   [-1., 0., 1.]])

	points[:, :-1] *= R * 0.5
	points[:, -1] *= height * 0.5#	0.5 надо т.к. начало координат не в (0,0), а в центре фигуры

	# построение призмы
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	prisms = []
	list_of_points = [points]
	approximate(list_of_points, alpha)
	for i in range(len(list_of_points)):
		prisms += [make_sides(list_of_points[i])]

	for i in range(len(prisms)):
		ax.add_collection3d(Poly3DCollection(prisms[i], linewidths=0., facecolors = choose_color(color, view), edgecolors='k'))

	# настройка графика
	ax.set_aspect("equal")
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	limit = height * 0.5 * 1.2
	plt.xlim([-limit, limit])
	plt.ylim([-limit, limit])
	ax.set_zlim(-limit, limit)

	plt.show()