#include <sys/wait.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

#define PIPE_READ 0
#define PIPE_WRITE 1

int fd[2]; 

int IsPrime(int number, int d)
{
    pid_t pid;
    int buf, res, status, bufread;
    printf("%d\n", d);
    if (number < 2) {
        //printf("No\n");
        return 0;
    } else if (d == 1) {
        //printf("Yes\n");
        return 1;
    }
    
    pid = fork();
    if (pid == 0) {
        if (number % d == 0) {
            buf = 0;
        } else {
            buf = IsPrime(number, d - 1);       
        }
        close(fd[PIPE_READ]);
        if(write(fd[PIPE_WRITE], &buf, sizeof(buf)) == -1) {
            perror("write");
        }
        exit(0);
    } else if (pid < 0) {
        perror("fork");
        exit(1);
    } else if (pid > 0) {
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
        }
    }

    if(read(fd[PIPE_READ], &bufread, sizeof(bufread)) == -1) {
        perror("read");
    }
    res = bufread;

    return res;
    
}

int main(void) {
    int number = 0;
    
    if (pipe(fd) == -1) { //предоставляет средства передачи данных между двумя процессами
        perror("pipe");
    }
    
    printf("Enter any positive number:");
    scanf("%d", &number);
    
    if (number < 0) {
        printf("Number must be > 0.\n");
    } else {
        if(IsPrime(number, number / 2))
            printf("Yes\n");
        else
            printf("No\n");
    }
}
