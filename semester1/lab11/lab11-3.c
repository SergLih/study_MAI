#include <stdio.h>

#define FALSE 0
#define TRUE 1
#define BOOL int
#define OTHER 0
#define SPACE 1
#define TRY_GET_RESULT 2
#define MY_LONG_MAX 100000000000000000

typedef struct
{
    unsigned long first;
    unsigned long second;
    BOOL is_have_second;
} Buffer;

int is_empty(Buffer buffer);
BOOL is_have_over_one_symbol(Buffer buffer);

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

int is_lexical_order(Buffer buffer, char symbol)
{
    unsigned long last_symbol = !buffer.is_have_second? buffer.first % 100 : buffer.second % 100;
    if (last_symbol == 'a') {
        last_symbol = 'A';
    }
    return last_symbol <= (unsigned long)symbol;
}

void print_result_long(unsigned long buffer)
{
    if (buffer == 0) {
        return;
    }
    
    if (buffer % 100 == 'a') {
        buffer = buffer - ('a' - 'A');
    }
    
    print_result_long(buffer / 100);
    printf("%c", (char)(buffer % 100));
}

void print_result(Buffer buffer)
{
    print_result_long(buffer.first);
    print_result_long(buffer.second);
}

int is_end(Buffer buffer, char symbol)
{
    if (!delimiter(symbol)) {
        return FALSE;
    }
    if (is_have_over_one_symbol(buffer)) {
        return TRUE;
    }
    return buffer.first != (unsigned long)'-' && buffer.second != (unsigned long)'+';
}

int is_empty(Buffer buffer)
{
    return buffer.first > 0;
}

BOOL is_have_over_one_symbol(Buffer buffer)
{
    return buffer.first > 100;
} 

void clear(Buffer* buffer)
{
    buffer->first = 0;
    buffer->second = 0;
    buffer->is_have_second = FALSE;
}

void add(Buffer* buffer, char symbol)
{
    if(buffer->first < MY_LONG_MAX)
        buffer->first = buffer->first * 100 + (unsigned long)symbol;
    else {
        buffer->is_have_second = TRUE;
        buffer->second = buffer->second * 100 + (unsigned long)symbol;
    }
}

int main(void)
{
    Buffer buffer;
    clear(&buffer);
    int is_result_printed = FALSE;
    int state = SPACE;
    char symbol;
    while (scanf("%c", &symbol) != EOF) {
        switch (state) {
            case OTHER:
                clear(&buffer);
                if (delimiter(symbol)) {
                    state = SPACE;
                }
                break;

            case SPACE:
                if (numbers11(symbol) || symbol == '-' || symbol == '+') {
                    state = TRY_GET_RESULT;
                    clear(&buffer);
                    add(&buffer, symbol);
                } else {
                    state = OTHER;
                }
                break;

            case TRY_GET_RESULT:
                if (is_end(buffer, symbol)) {
                    print_result(buffer);
                    is_result_printed = TRUE;
                    clear(&buffer);
                    state = SPACE;
                    printf(" ");
                } else if (!numbers11(symbol) || !is_lexical_order(buffer, symbol)) {
                    if (delimiter(symbol)) {
                        state = SPACE;
                    } else {
                        state = OTHER;
                    }
                } else {
                    add(&buffer, symbol);
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
