#ifndef _STUDENT_H_
#define _STUDENT_H_

#include "class.h"

void generate_student(Student * student);
StudArr * create_students(size_t n);
void print_students(StudArr * students);
void destroy_students(StudArr ** students);

#endif
