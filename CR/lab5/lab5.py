import urllib
import string
import random
import re
import numpy as np
import matplotlib.pyplot as plt

def gen_random_words(n):
    with open('russian_words.txt', 'r') as f:
        list_words = [word for word in f]
    del list_words[0:4]
    words_res = random.choices(list_words, k=n)
    return ''.join(words_res).replace('\n', '').replace(' ', '').replace('-', \
                                                   '').lower().replace('ё', 'е')

def gen_random_chars(n):
    rus_alp = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    char_rus = list(map(lambda x: x.lower(), list(rus_alp)))
    return ''.join(random.choice(char_rus) for i in range(n))

def count_frequencies(text):
    alpha = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
    return [text.count(a)/len(text) for a in alpha]
    
def count_common_letters(text1, text2):
    return sum(text1[i]==text2[i] for i in range(min(len(text1), len(text2))))
    
def count_common_letters_array(text1, text2):
    min_len = min(len(text1), len(text2))
    return np.cumsum([text1[i]==text2[i] for i in range(min_len)])/np.arange(1, min_len+1)
    
def letter_match_percentage(text1, text2):
    return count_common_letters(text1, text2) / len(text1)

   
# Два осмысленных текста   
 
   
content1 = urllib.request.urlopen('http://www.rusf.ru/abs/books/tbb100.htm').\
                                                read().decode('Windows-1251')
res_list1 = re.findall(r'[А-Яа-я]', content1)
text1 = ''.join(res_list1).lower()
content2 = urllib.request.urlopen('http://lib.ru/FOUNDATION/f_voyage.txt')\
                                                    .read().decode('koi8-r')
res_list2 = re.findall(r'[А-Яа-я]', content2)
text2 = ''.join(res_list2).lower()
min_len = min(len(text1), len(text2))
text1 = text1[:min_len]
text2 = text2[:min_len]
print("1) Two meaningful texts:\n")
print(text1[:100])
print(text2[:100])
print("\nText length: {0}".format(min_len))
print("Percentage: {0}".format(letter_match_percentage(text1, text2)))

#f1 = np.array(count_frequencies(text1))
#f2 = np.array(count_frequencies(text2))
#f3 = np.array(count_frequencies(text1+text2))

#print(np.sum(f1**2))
#print(np.sum(f2**2))
#print(np.sum(f3**2))

plt.figure(figsize=(10, 7))
plt.title('Два осмысленных текста')
plt.plot(count_common_letters_array(text1, text2), label='% совпадений')
plt.axhline(0.05521977904636903, color='red', alpha=0.65, label='ожидаемый \
                                    индекс совпадений\nдля данных текстов = 0.055219')
plt.axvline(10000, color='gray', alpha=0.65, label='длина текстов = 10000')
plt.ylim((0.025, 0.065))
plt.legend(loc='center right')
plt.savefig('Two_meaningful_texts')


#Осмысленный текст и текст из случайных букв


print("\n2) Meaningful text and text from random chars:\n")
print(text1[:100])
text2 = gen_random_chars(len(text1))
print(text2[:100])
print("\nText length: {0}".format(len(text1)))
print("Percentage: {0}".format(letter_match_percentage(text1, text2)))


plt.figure(figsize=(10, 7))
plt.title('Осмысленный текст и текст из случайных букв')
plt.plot(count_common_letters_array(text1, text2), label='% совпадений')
plt.axhline(1/32, color='red', alpha=0.65, label='ожидаемый индекс совпадений\n\
                                                        для данных текстов = 0.03125')
plt.axvline(20000, color='gray', alpha=0.65, label='длина текстов = 20000')
plt.ylim((0.015, 0.07))
plt.legend(loc='center right')
plt.savefig('Meaningful_text_and_text_from_random_chars')


#Осмысленный текст и текст из случайных слов


print("\n3) Meaningful text and text from random words:\n")
print(text1[:100])
text2 = gen_random_words(len(text1))
print(text2[:100])
print("\nText length: {0}".format(len(text1)))
print("Percentage: {0}".format(letter_match_percentage(text1, text2)))


plt.figure(figsize=(10, 7))
plt.title('Осмысленный текст и текст из случайных слов')
plt.plot(count_common_letters_array(text1, text2), label='% совпадений')
plt.axhline(0.05484232566135076, color='red', alpha=0.65, label='ожидаемый индекс \
                                        совпадений\nдля данных текстов = 0.054842')
plt.axvline(80000, color='gray', alpha=0.65, label='длина текстов = 80000')
plt.ylim((0.025, 0.065))
plt.legend(loc='center right')
plt.savefig('Meaningful_text_and_text_from_random_words')


#Два текста из случайных букв


size = len(text1)
text1 = gen_random_chars(size)
text2 = gen_random_chars(size)
print("\n4) Two texts from random chars:\n")
print(text1[:100])
print(text2[:100])
print("\nText length: {0}".format(size))
print("Percentage: {0}".format(letter_match_percentage(text1, text2)))


plt.figure(figsize=(10, 7))
plt.title('Два текста из случайных букв')
plt.plot(count_common_letters_array(text1, text2), label='% совпадений')
plt.axhline(1/32, color='red', alpha=0.65, label='ожидаемый индекс совпадений\n\
                                                    для данных текстов = 0.03125')
plt.axvline(30000, color='gray', alpha=0.65, label='длина текстов = 30000')
plt.ylim((0.015, 0.11))
plt.legend(loc='center right')
plt.savefig('Two_texts_from_random_chars')


#Два текста из случайных слов


text1 = gen_random_words(min_len//3)[:min_len]
text2 = gen_random_words(min_len//3)[:min_len]
print("\n5) Two texts from random words:\n")
print(text1[:100])
print(text2[:100])
print("\nText length: {0}".format(size))
print("Percentage: {0}".format(letter_match_percentage(text1, text2)))


plt.figure(figsize=(10, 7))
plt.title('Два текста из случайных слов')
plt.plot(count_common_letters_array(text1, text2), label='% совпадений')
plt.axhline(0.05349873904539696, color='red', alpha=0.65, label='ожидаемый индекс \
                                        совпадений\nдля данных текстов = 0.053498')
plt.axvline(50000, color='gray', alpha=0.65, label='длина текстов = 50000')
plt.ylim((0.025, 0.08))
plt.legend(loc='upper right')
plt.savefig('Two_texts_from_random_words')

