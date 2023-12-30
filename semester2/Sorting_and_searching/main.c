#include <stdio.h>
#include <stdlib.h>

#include "table.h"
#include "key.h"

int main(void)
{
    srand((size_t)time(0));
    size_t rows = 0;
    size_t pos;
    Table table = table_create();
    Key key = key_create();

    char s[9];
    unsigned int search_time;

    printf("\nДля получения помощи в использовании программы напишите help или h.\n\n");
    while (1) {
        scanf("%s", s);
        if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0 || strcmp(s, "q") == 0) {
            if (table != NULL) {
                table_destroy(&table);
            }
            key_destroy(&key);
            break;
        } else if (strcmp(s, "print") == 0 || strcmp(s, "p") == 0) {
            if (table == NULL)
                printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else
                table_print(table);
        } else if (strcmp(s, "insert") == 0 || strcmp(s, "ins") == 0) {
            if(table == NULL)
                table = table_create();
            rows = -1;
            while(scanf("%zd", &rows) != 1)	{
                printf("Ошибка. Введите корректное число строк:\n");	
                clean_stdin();
            }
            for(int j = 0; j < rows; j++) {
                table_input_and_push(table);
            }
        } else if (strcmp(s, "delete") == 0 || strcmp(s, "del") == 0) {
            if(table == NULL) printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else {
                key = key_input(false);
                table_delete(table, key);
            }
        } else if (strcmp(s, "sort") == 0 || strcmp(s, "s") == 0) {
            if(table == NULL)
                printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else
                table_sort(table);
        }  else if (strcmp(s, "find") == 0 || strcmp(s, "f") == 0) {
            if (table == NULL) printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else {
                key = key_input(false);
                Value v = table_search(table, key, &pos);
                if (v == NULL) {
                    printf("Записи с таким ключом нет в таблице.\n");
                } else {
                    printf("Найдена запись под порядковым номером %zd:\n", pos);
                    row_print(table, pos);
                }
            }
        } else if (strcmp(s, "clear") == 0 || strcmp(s, "c") == 0) {
            if (table == NULL) printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else {
                table_destroy(&table);
                table = table_create();
            }
        } else if (strcmp(s, "reverse") == 0 || strcmp(s, "rev") == 0) {
            if (table == NULL) printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else {
                table_reverse(table);
            }
        } else if (strcmp(s, "shuffle") == 0 || strcmp(s, "sh") == 0) {
            if (table == NULL) printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else {
                table_shuffle(table);
            }
        } else if (strcmp(s, "generate") == 0 || strcmp(s, "g") == 0) {
            if (table == NULL) printf("Таблицы не существует, воспользуйтесь командами help или h.\n");
            else {
                scanf("%zd", &rows);
                size_t len = 70;
                size_t rem = rows % len;
                printf("Generation in progress... ");
                for(size_t i = 0; i < len; i++) {
                    for(size_t j = 0; j < rows/len; j++)
                        generate_row(i, len, table);
                    }
                for(size_t i = 0; i < rem; i++)
                    generate_row((len-1), len, table);
                printf("done!\n");
                //table_shuffle(table);
            }
        }

        else if (strcmp(s, "help") == 0 || strcmp(s, "h") == 0) {
          printf("\n\nins[ert] <rows>  добавить <rows> строк в таблицу");
            printf("\np[rint]          печатать таблицу");
            printf("\ns[ort]           сортировать таблицу по ключам рекурсивной сортировкой Хоара");
            printf("\nf[ind]   <key>   найти в таблице элемент с ключом <key>");
            printf("\ndel[ete] <key>   удалить из таблицы элемент с ключом <key>");
            printf("\nrev[erse]        переставить строки таблицы в обратном порядке");
            printf("\nsh[uffle]        перемешать строки таблицы в случайном порядке");
            printf("\ng[enerate] <n>   добавить в таблицу <n> случайных строк");
            printf("\nc[lear]          очистить таблицу");
            printf("\nq[uit]           выйти из программы\n\n");
        }
    }
    return 0;
}
