# Фамилия: Лихарев С.С.
# Группа:  M80-307Б
# Условие задачи:  Написать и отладить программу, строящую изображение заданной замечательной кривой
# Вариант задания: 09. ρ = a * ϕ, 0 ≤ ϕ ≤ B

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
 
# начальные условия
a0 = 1
B0 = 6
a = a0
B = B0
# массив возможных значений r
r = np.arange(0, B, 0.01)
# тот же массив, но умноженный на пи
phi = np.pi * r
# значение функции на этом диапазоне
rho = a * phi
 
# построение графика, настройка отступов для элементов управления
fig = plt.figure()
ax = plt.subplot(111, projection='polar')
plt.subplots_adjust(bottom=0.25)
 
# построение линии
l, = ax.plot(phi, rho, linewidth=3)
# вывод формулы на окне программы
ax.text(0, 1, r'$\rho = a * \varphi,\ 0\,\leqslant \,\varphi\, \leqslant\, B$', horizontalalignment='center',
 	verticalalignment='center',
 	transform=ax.transAxes)
 
# настройка осей и подписей на них
ax.set_rmax(20)
ax.set_rticks(list(range(5, 20, 5)))
ax.set_rlabel_position(-22.5)
ax.grid(True)
 
# специальные оси для слайдеров (их позиция)
axcolor = 'lightgoldenrodyellow'
ax_a = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_B = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
 
# создание слайдеров
slider_B = Slider(ax_B, 'B', 0.1, 10.0, valinit=B0, valstep=0.1, valfmt='%.1f$\pi$')
slider_a = Slider(ax_a, 'a', 0.5, 10.0, valinit=a0, valstep=0.5)
 
# функция, которая обновляет массивы после движения слайдеров
def update(val):
	a = slider_a.val
	B = slider_B.val
	r = np.arange(0, B, 0.01)
	phi = np.pi * r
	rho = a * phi
	l.set_xdata(phi)
	l.set_ydata(rho)
	fig.canvas.draw_idle()
 
# привязываем функцию update событию изменения слайдеров	
slider_a.on_changed(update)
slider_B.on_changed(update)
 
# создание кнопки Reset
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
 
# функция, которая отвечает за сброс параметров на значения по умолчанию
def reset(event):
	slider_a.reset()
	slider_B.reset()
button.on_clicked(reset)
 
plt.show()
