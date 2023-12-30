#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include <inttypes.h>

typedef uint32_t Set;

#define EMPTY_SET ((Set) 0UL)

int delimiter(char c)
{
    return c == ' ' ||
           c == ',' ||
           c == '\t' ||
           c == '\n' ||
           c == '\r' ||
           c == EOF;
}

Set set_add(Set s, unsigned int c)
{
    return s |= 1u << c;
}

Set set_intersection(Set s1, Set s2)
{
    return s1 & s2;
}
 
Set set_union(Set s1, Set s2)
{
    return s1 | s2;
}

bool is_empty(Set s)
{
    return s == EMPTY_SET;
}

bool sets_equal(Set s1, Set s2)
{
    return set_intersection(s1, s2) == set_union(s1, s2);
}

int main(void)
{
    Set curr = EMPTY_SET;
    Set prev = EMPTY_SET;
    unsigned int c = 0;
    bool have_result = false;
    Set result = 0;
    while (true) {
        c = getchar();
        if (delimiter(c)) {
            if (!is_empty(curr) && !is_empty(prev)) {
                have_result = true;
            }
            if (!is_empty(curr)) {
                prev = curr;
                curr = EMPTY_SET;
            }
        } else {
            curr = set_add(curr, c);
        }
        if (c == EOF) {
            break;
        }
    }
    if (!have_result) {
        printf("No\n");
    }
    if (have_result == true) {
       printf(sets_equal(curr, prev) ? "No\n" : "Yes\n");
    }
    return 0;
}

