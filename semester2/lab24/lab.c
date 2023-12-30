#include <stdio.h>

#include "lab.h"
#include "printer.h"

int count_minuses(Expression expr) {
    if (expr == NULL) {
        return 0;
    }

    if(expr->arity != TERMINAL)
    {
        int count = 0;
        if (expr->arity == UNARY && expr->data.type == OPERATOR && expr->data.data.operator_name == '-')
             count += 1;
        count += count_minuses(expr->left) + count_minuses(expr->right);
        return count;
    }
    return 0;
}

void expression_create_unary_minus(Expression * u_minus_expr, Expression expr_under_minus)
{
    Token * u_minus_token = (Token*) malloc(sizeof(Token));
    u_minus_token->type = OPERATOR;
    u_minus_token->data.operator_name = '-';
    expression_create_unary(u_minus_expr, u_minus_token, expr_under_minus);
}

void transform_minuses(Expression expr) {
    if (expr == NULL) {
        return;
    }
    if (expr->arity != TERMINAL) {
        Expression e_left = expr->left;
        Expression e_right = expr->right;

        if (e_left->arity == TERMINAL) {
            Expression u_minus_expr;
            if (e_left->data.type == INTEGER && e_left->data.data.value_int < 0) {
                e_left->data.data.value_int *= (-1);
                expression_create_unary_minus(&u_minus_expr, e_left);
                expr->left = u_minus_expr;
            }
            if (e_left->data.type == FLOATING && e_left->data.data.value_float < 0) {
                e_left->data.data.value_float *= (-1);
                expression_create_unary_minus(&u_minus_expr, e_left);
                expr->left = u_minus_expr;
            }
        } else {
            transform_minuses(e_left);
        }
        if(e_right != NULL) {
            if (e_right->arity == TERMINAL) {
                Expression u_minus_expr;
                if (e_right->data.type == INTEGER && e_right->data.data.value_int < 0) {
                    e_right->data.data.value_int *= (-1);
                    expression_create_unary_minus(&u_minus_expr, e_right);
                    expr->right = u_minus_expr;
                }
                if (e_right->data.type == FLOATING && e_right->data.data.value_float < 0) {
                    e_right->data.data.value_float *= (-1);
                    expression_create_unary_minus(&u_minus_expr, e_right);
                    expr->right = u_minus_expr;
                }
            } else {
                transform_minuses(e_right);
            }
        }
    }

}

void delete_minuses_r(Expression *expr)
{
    if (*expr == NULL) {
        return;
    }

    if((*expr)->arity != TERMINAL)
    {
        //ydalyaem unarnue minusu i skobki pod kotorumi ne binarnue operatoru
        while((*expr)->left->arity == UNARY && (*expr)->left->left->arity != BINARY)
        {
            Expression expr_under_op = (*expr)->left->left;
            token_free(&(((*expr)->left)->data));
            free((*expr)->left);
            (*expr)->left = expr_under_op;
        }
        if((*expr)->right != NULL)
        {
            while((*expr)->right->arity == UNARY && (*expr)->right->left->arity != BINARY)
            {
                Expression expr_under_op = (*expr)->right->left;
                token_free(&(((*expr)->right)->data));
                free((*expr)->right);
                (*expr)->right = expr_under_op;
            }
        }
        delete_minuses_r(&((*expr)->left));
        delete_minuses_r(&((*expr)->right));
    }

}

void delete_minuses(Expression * root_expr)
{
    Expression b_expr;
    Token b_token;
    b_token.type = BRACKET;
    b_token.data.is_left_bracket = true;
    expression_create_unary(&b_expr, &b_token, *root_expr);
    *root_expr = b_expr; //delaem novyu vershinu kornem

    int k_minuses = count_minuses(*root_expr);
    delete_minuses_r(root_expr);

    if(k_minuses % 2 == 0)
    {
        Expression expr_under_op = (*root_expr)->left;
        token_free(&((*root_expr)->data));
        free(*root_expr);
        *root_expr = expr_under_op;
    }
    else
    {
        (*root_expr)->data.type = OPERATOR;
        (*root_expr)->data.data.operator_name = '-';
    }
}

void expression_print_infix(Expression expr)
{
    expression_print_infix_r(expr);
    printf("\n");
}

void expression_print_infix_r(Expression expr)
{
    if(expr == NULL)
        return;

    switch (expr->arity) {
        case BINARY:
            //printf("(");
            expression_print_infix_r(expr->left);
            token_print(&(expr->data));
            expression_print_infix_r(expr->right);
            //printf(")");
            break;
        case UNARY:
            if (expr->data.type == OPERATOR) {
                token_print(&(expr->data));
                expression_print_infix_r(expr->left);
            } else {
                printf("(");
                expression_print_infix_r(expr->left);
                printf(")");
            }
            break;
        case TERMINAL:
            token_print(&(expr->data));
            break;
    }
}
