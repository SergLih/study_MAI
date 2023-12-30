#include <stdio.h>

#define FALSE 0
#define TRUE 1
#define OTHER 0
#define SPACE 1
#define TRY_GET_RESULT 2

int delimiter(char c)
{
    return c == ' ' ||
           c == ',' ||
           c == '\t' ||
           c == '\n' ||
           c == '\r';
}

int numbers11(char c)
{
    return (c >= '0' && c <= '9') || c == 'A' || c == 'a';
}

int is_lexical_order(unsigned long buffer, char symbol)
{
    unsigned long last_symbol = buffer % 100;
    if (last_symbol == 'a') {
        last_symbol = 'A';
        return last_symbol <= (unsigned long)symbol;
    }
    
    return buffer % 100 <= (unsigned long)symbol;
}

void print_result(unsigned long buffer)
{
    if (buffer == 0) {
        return;
    }
    
    if (buffer % 100 == 97) {
        buffer = buffer - 32;
    }
    
    print_result(buffer / 100);
    printf("%c", (char)(buffer % 100));
}

int is_end(unsigned long buffer, char symbol)
{
    if (!delimiter(symbol)) {
        return FALSE;
    }
    if (buffer > 100) {
        return TRUE;
    }
    return buffer != (unsigned long)'-' && buffer != (unsigned long)'+';
}

int main(void)
{
    unsigned long buffer = 0;
    int is_result_printed = FALSE;
    int state = SPACE;
    char symbol;
    while (scanf("%c", &symbol) == 1) {
        switch (state) {
            case OTHER:
                buffer = 0;
                if (delimiter(symbol)) {
                    state = SPACE;
                }
                break;

            case SPACE:
                if (numbers11(symbol) || symbol == '-' || symbol == '+') {
                    state = TRY_GET_RESULT;
                    buffer = (unsigned long)symbol;
                } else {
                    state = OTHER;
                }
                break;

            case TRY_GET_RESULT:
                if (is_end(buffer, symbol)) {
                    print_result(buffer);
                    is_result_printed = TRUE;
                    state = SPACE;
                    printf(" ");
                } else if (!numbers11(symbol) || !is_lexical_order(buffer, symbol)) {
                    if (delimiter(symbol)) {
                        state = SPACE;
                    } else {
                        state = OTHER;
                    }
                } else {
                    buffer = buffer * 100 + (unsigned long)symbol;
                }
                break;
        }
        if (symbol == '\n' && is_result_printed) {
            is_result_printed = FALSE;
            printf("\n");
        }
    }
    return 0;
}