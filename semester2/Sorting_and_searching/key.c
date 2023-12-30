#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "table.h"
#include "key.h"

int key_compare(Key k1, Key k2)        //сравнение ключей, реализация для float
{
    if (k1 < k2)
        return -1;
    else if (k1 > k2)
        return 1;
    else
        return 0;
}

void key_copy(Key * dest, Key * src)        //копирование значений - куда, откуда
{
    *dest = *src;                           //реализация для float
}

Key key_input(bool prompt)             //ввод ключа с клавиатуры, реализация для float
{
    Key k;
    char sym;
    if(prompt)
        printf("Введите ключ:\n");
    while (scanf("%f%c", &k, &sym) != 2) {
        printf("Ошибка. Некорректный ключ \\_(о_О')_/ Попробуйте ещё разок: \n");
        clean_stdin();
    } 
    return k;
}

void value_copy(Value * dest, Value * src)  //копирование значений - куда, откуда
{
    size_t len = strlen(*src);
    value_destroy(dest);
    *dest = (Value) malloc(sizeof(char)*(len+1));   //реализация для си-строк

    strncpy(*dest, *src, len);
    (*dest)[len] = '\0';
}

Value value_input()
{
    Value v = NULL;
    char sym;
    printf("Введите строку ASCII-картинки:\n");
    char * buf = (char*) malloc(VALLEN*sizeof(char)); //
    size_t j = 0;
    while ((sym = getchar()) != '\n' && j < VALLEN) {
        buf[j++] = sym;
    }
    buf[j++] = '\0';
    value_copy(&v, (Value *)(&buf));
    free(buf);
    return v;
}

Key key_create()                            //создание ключа
{	
    return 0;                               //реализация для float
}

void key_destroy(Key *k)                    //освобождение памяти ключа
{
    return;                                 //реализация для float
}

void value_destroy(Value * v)               //освобождение памяти значения
{
    if(*v != NULL) {
        free(*v);                           //реализация для си-строк
        *v = NULL;
    }
}
