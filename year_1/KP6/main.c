#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "api.h"

void help()
{
	printf(
		"-h, --help        Вывод справки\n"
		"<dbname> -v       Вывод базы из файла <dbname> на экран\n"
		"<dbname> -g <n>   Сгенерировать базу из <n> записей и сохранить в файле <dbname>\n"
		"<dbname> -q       Вывести результат запроса к базе в файле <dbname>\n"
		);
}

uint32_t main(uint32_t argc, char **argv)
{
    StudArr * sts = NULL;
    srand(time(NULL));
    printf("%d\n", argc);
    if (argc < 2) {                     //первый аргумент всегда имя программы, поэтому проверяем их 2 хотя бы
        printf("Недостаточно аргументов. Для получения справки запустите с параметром -h(--help)");
        exit(0);
    }
    
    if(strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        help();
        exit(0);
    } 

    //generate
    if(strcmp(argv[2], "-g") == 0) {
        if(argc != 4) {
            printf("Недостаточно аргументов. Для получения справки запустите с параметром -h(--help)");
            exit(0);
        }
        size_t n = (size_t) atoi(argv[3]);
        printf("n=%zd\n", n);
        sts = create_students(n);
        for(size_t i = 0; i < sts->size; i++){
            generate_student(&(sts->st_arr[i]));
        }
        FILE * f = fopen(argv[1], "wb");
        if(f){
            fwrite(sts->st_arr, sizeof(Student), sts->size, f);
            fclose(f);
            printf("Файл успешно записан!\n");
        } else {
            printf("Ошибка. Невозможно открыть файл для записи!\n");
        }
    }

    //load and print
    if(strcmp(argv[2], "-v") == 0) {
        if(argc != 3) {
            printf("Недостаточно аргументов. Для получения справки запустите с параметром -h(--help)");
            exit(0);
        }

        FILE * f = fopen(argv[1], "rb");
        if(f){
            fseek (f , 0 , SEEK_END);
            size_t n = ftell (f) / sizeof(Student);
            rewind (f);

            sts = create_students(n);

            size_t result = fread(sts->st_arr, sizeof(Student), n, f);
            if (result != n) {
                printf ("Reading error: only %zd out of %zd elements read\n", result, n);
                exit (3);
            }
            print_students(sts);
        } else {
            printf("Ошибка. Невозможно открыть файл!\n");
        }
    }

    //load and print
    if(strcmp(argv[2], "-q") == 0) {
        if(argc != 3) {
            printf("Недостаточно аргументов. Для получения справки запустите с параметром -h(--help)");
            exit(0);
        }

        FILE * f = fopen(argv[1], "rb");
        if(f){
            fseek (f , 0 , SEEK_END);
            size_t n = ftell (f) / sizeof(Student);
            rewind (f);

            sts = create_students(n);

            size_t result = fread(sts->st_arr, sizeof(Student), n, f);
            if (result != n) {
                printf ("Reading error: only %zd out of %zd elements read\n", result, n);
                exit (3);
            }
            query_students(sts);
        } else {
            printf("Ошибка. Невозможно открыть файл!\n");
        }
    }
    destroy_students(&sts);
    return 0;
}
