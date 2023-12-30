#ifndef _CLASS_H_
#define _CLASS_H_

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>

#define STR_SIZE 32
#define MAX_CLASSES 11
#define MAIN_CLASS 10

typedef enum GENDER {
    Male, Female
} Gender;

typedef struct {
    char surname[STR_SIZE];
    char initials[STR_SIZE];
    Gender gender;
    int school_number;
    int math;
    int rus;
    int prof;
    bool essay;
    bool has_medal;
} Student;

typedef struct {
    size_t   size;
    Student *st_arr;
} StudArr;

#endif
