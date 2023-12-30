#ifndef LAB_H
#define LAB_H
#include "expression.h"

int count_minuses(Expression expr);

void expression_create_unary_minus(Expression * u_minus_expr, Expression expr_under_minus);

void transform_minuses(Expression expr);

void delete_minuses_r(Expression *expr);

void delete_minuses(Expression * root_expr);

void expression_print_infix(Expression expr);
void expression_print_infix_r(Expression expr);

#endif // LAB_H
