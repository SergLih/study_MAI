#include <stdio.h>

#define FALSE 0
#define TRUE 1
#define OTHER 0
#define SPACE 1
#define TRY_GET_RESULT 2

int is_empty(unsigned long buffer);

int delimiter(char c)
{
    return c == ' ' ||
           c == ',' ||
           c == '\t' ||
           c == '\n' ||
           c == '\r' ||
           c == EOF;
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
    
    if (buffer % 100 == 'a') {
        buffer = buffer - ('a' - 'A');
    }
    
    print_result(buffer / 100);
    printf("%c", (char)(buffer % 100));
}

int is_end(unsigned long buffer, char symbol)
{
    if (!delimiter(symbol)) {
        return FALSE;
    }
    if (!is_empty(buffer)) {
        return TRUE;
    }
    return buffer != (unsigned long)'-' && buffer != (unsigned long)'+';
}

int is_empty(unsigned long buffer)
{
    return buffer < 100;
}

int main(void)
{
    unsigned long buffer = 0;
    int is_result_printed = FALSE;
    int state = SPACE;
    char symbol;
    while (scanf("%c", &symbol) != EOF) {
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
                    buffer = 0;
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
    if (!is_empty(buffer)) {
        print_result(buffer);
        printf("\n");
    }
    return 0;
}
