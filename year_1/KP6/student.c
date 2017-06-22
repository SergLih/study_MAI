#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api.h"

int randint(int a, int b)
{
    return rand() % (b-a) + a;
}

int randbool()
{
    return rand()%2 ? true : false;
}

void print_student(Student * st)
{
    printf("%-12s %s | Gender: %c | School #%4d | Math: %d | Rus: %d | Prof: %d\n",
           st->surname, st->initials, (st->gender == Male ? 'M' : 'F'),
           st->school_number, st->math, st->rus, st->prof);
}

void generate_initials(char s[STR_SIZE])
{
    s[0] = randint('A', 'Z'+1);
    s[1] = '.';
    s[2] = randint('A', 'Z'+1);
    s[3] = '.';
    s[4] = '\0';
}

void generate_surname(char s[STR_SIZE], Gender *gender)
{
    char * syl = "kokakutatetotutibabebobuneninanozazugogagupopapipumomamemurorarireru";
    char * end = {"navaevinnko"};
    int syl_len = 33;
    int end_len = 5;

    int syl_n = randint(2, 4);
    int L = 0;
    for(int i = 0; i<syl_n; i++){
        int syl_i = randint(0, syl_len);
        for(int j = 0; j < 2; j++)
            s[L++] = syl[syl_i*2 + j];
    }
    int end_i = randint(0, end_len);
    for(int j = 0; j < (end_i == 4? 3 : 2); j++)
        s[L++] = end[end_i*2 + j];
    s[L] = '\0';

    s[0] = toupper(s[0]);

    if(end_i < 2)
        *gender = Female;
    else
        *gender = Male;
}

void generate_student(Student * student)
{
    Gender gender;
    generate_surname(student->surname, &gender);
    generate_initials(student->initials);
    student->gender = gender;
    student->essay = randbool();
    student->has_medal = randbool();
    student->school_number = randint(0, 10000);
    student->math = randint(3,6);
    student->rus  = randint(3,6);
    student->prof = randint(3,6);
}

StudArr * create_students(size_t n)
{
    StudArr * students = (StudArr *) malloc(sizeof(StudArr));
    students->size = n;
    students->st_arr = (Student *) malloc(n*sizeof(Student));
    return students;
}

void print_students(StudArr * students)
{
    for(size_t i = 0; i < students->size; i++){
        print_student(&(students->st_arr[i]));
    }
}

void destroy_students(StudArr ** students)
{
    if(*students == NULL)
        return;
    free((*students)->st_arr);
    free(*students);
    *students = NULL;
}

void query_students(StudArr * st){
    for(size_t i = 0; i < st->size; i++){
            if(!(    st->st_arr[i].math == st->st_arr[i].rus
                  || st->st_arr[i].math == st->st_arr[i].prof
                  || st->st_arr[i].rus == st->st_arr[i].prof)) {
                print_student(&(st->st_arr[i]));
            }
    }
}
