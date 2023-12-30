#Диалоговое окно

from tkinter import *
import tkinter.filedialog
from lab_4_5 import main

#функция загрузки файла
def LoadFile(event): 
	fn = tkinter.filedialog.Open(root, filetypes = [('*.txt files', '.txt')]).show()
	if fn == '':
		return
	inp_a.delete(0, END)
	inp_h.delete(0, END)
	inp_alpha.delete(0, END)

	inp_a.insert(0, open(fn, 'r').read().split()[0])
	inp_h.insert(0, open(fn, 'r').read().split()[1])
	inp_alpha.insert(0, open(fn, 'r').read().split()[2])

#функция записи файла
def SaveFile(event):
	fn = tkinter.filedialog.SaveAs(root, filetypes = [('*.txt files', '.txt')]).show()
	if fn == '':
		return
	if not fn.endswith(".txt"):
		fn+=".txt"
	open(fn, 'w').write(inp_a.get() + ' '+ inp_h.get() + ' ' + inp_alpha.get())

#функция построения призмы
def Build(event):
	main(float(inp_a.get()), float(inp_h.get()), int(inp_alpha.get()))


root = Tk()
root.geometry('+150+150')

#справочник по использованию диалогового окна
man = Label(text = "Введите высоту, длину стороны и точность аппроксимации.", font = "Arial 10")
man.pack(side = "top")


#создание кнопок и их настройки
panelFrame = Frame(root, height = 30, bg = "gray")
panelFrame.pack(side = "bottom", fill = 'x')

draw = Button(panelFrame, text = "Построить")
save = Button(panelFrame, text = 'Сохранить')
load = Button(panelFrame, text = 'Загрузить')

draw.bind("<Button-1>", Build)
save.bind("<Button-1>", SaveFile)
load.bind("<Button-1>", LoadFile)

draw.place(x = 10, y = 6, width = 100, height = 22)
save.place(x = 120, y = 6, width = 100, height = 22)
load.place(x = 230, y = 6, width = 100, height = 22)

#оформление рамки

textFrame = Frame(root)
textFrame.pack(side = "top", fill = "both", expand = 1)

inp_a = Entry(textFrame, width = 20)
inp_a.insert(0, "сторона")

inp_h = Entry(textFrame, width = 20)
inp_h.insert(0, "высота")

inp_alpha = Entry(textFrame, width = 20)
inp_alpha.insert(0, "точность")

inp_a.pack(side = 'left', fill = "both", expand = 1)
inp_h.pack(side = 'left', fill = "both", expand = 1)
inp_alpha.pack(side = 'left', fill = "both", expand = 1)


root.mainloop()