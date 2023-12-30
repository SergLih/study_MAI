from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
import subprocess as s
 
def choose_xml():
    sv_xml.set(fd.askopenfilename(filetypes=(("XML files", "*.xml"),)))
    conforms_to_xsd = False

def choose_xsd():
    sv_xsd.set(fd.askopenfilename(filetypes=(("XSD files", "*.xsd"),)))
    conforms_to_xsd = False

def choose_xsl():
    sv_xsl.set(fd.askopenfilename(filetypes=(("XSL files", "*.xsl"),)))
 
def choose_html():
    sv_html.set(fd.asksaveasfilename(
        filetypes=(("HTML files", "*.html;*.htm"),
                   ("All files", "*.*"))))
 
def check():
	if not sv_xml.get() or not sv_xsd.get():
		messagebox.showwarning(title='Ошибка', message='Не задан один из двух обязательных аргументов')
		return

	command = ['xmllint', '--schema', sv_xsd.get(), sv_xml.get(), '--noout']
	result = s.run(command, stdout=s.PIPE, stderr=s.PIPE, universal_newlines=True)

	if result.returncode == 0:
		messagebox.showinfo(title='', message='Проверка на соответствие XSD-схеме выполнена успешно!')
		conforms_to_xsd.set(True)
	else:
		messagebox.showwarning(title='Ошибка проверки схемы', message=result.stderr)


def transform():
	if not conforms_to_xsd.get():
		messagebox.showwarning(title='Ошибка', message='Документ не проверен на соответствие схеме')
		return

	if not sv_xml.get() or not sv_xsl.get() or not sv_html.get():
		messagebox.showwarning(title='Ошибка', message='Не задан один из обязательных аргументов')
		return

	if s.call(["xsltproc", "--output", sv_html.get(), sv_xsl.get(), sv_xml.get()]) == 0:
		messagebox.showinfo(title='', message='XSLT-преобразование выполнено успешно!')
	else:
		messagebox.showwarning(title='Ошибка', message='Что-то пошло не так :(')


 
root = Tk()
root.title('XSLT-преобразование')

sv_xml, sv_xsd, sv_xsl, sv_html = StringVar(), StringVar(), StringVar(), StringVar()
conforms_to_xsd = BooleanVar()

Label(text="XML-файл (данные):").grid(row=0, column=0)
Entry(width=50, textvariable=sv_xml).grid(row = 0, column=1)
Button(text="...", command=choose_xml).grid(row = 0, column=2)

Label(text="XSD-файл (схема): ").grid(row=1, column=0)
Entry(width=50, textvariable=sv_xsd).grid(row = 1, column=1)
Button(text="...", command=choose_xsd).grid(row = 1, column=2)

conforms_to_xsd.set(False)
Button(width=30, text='Проверить на соответствие схеме', command=check).grid(row=2, column=0, columnspan=3)

Label(text="XSL-файл (стиль): ").grid(row=3, column=0)
Entry(width=50, textvariable=sv_xsl).grid(row = 3, column=1)
Button(text="...", command=choose_xsl).grid(row = 3, column=2)

Label(text="HTML-файл (результат): ").grid(row=4, column=0)
Entry(width=50, textvariable=sv_html).grid(row = 4, column=1)
Button(text="...", command=choose_html).grid(row = 4, column=2)

Button(width=30, text='Преобразовать', command=transform).grid(row=5, column=0, columnspan=3)
 
root.mainloop()
