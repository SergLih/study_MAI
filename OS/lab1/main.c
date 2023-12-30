#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sendfile.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <inttypes.h>
#include <errno.h>

#define BUFFER_SIZE 256

int main(void) {
    char* str = "An experiment with a system call that writes a string to a file.\n";
    char buffer[BUFFER_SIZE];
    int fd_old, fd_new, w; 
    struct stat fileStat;
    char oldfilename[30];
    printf("Enter file name: ");
    scanf("%s", oldfilename);

    fd_old = creat(oldfilename, S_IREAD | S_IWRITE);
    if (fd_old == -1) {
        perror("creat()");
    } else {
        printf("File with filenasme '%s' was created \n", oldfilename);
    }

    if (w = write(fd_old, str, strlen(str)) == -1 ) {
        perror("write()");
    } 
    if (fsync(fd_old) == -1) {
        perror("fsync()");
    }
    
    if (close(fd_old) == -1) {
        perror("close()");
    }
    
    char dirname[30];
    printf("Enter the name of the folder to create: \n");
    scanf("%s", dirname);

    if (mkdir(dirname, 0755) == -1) {
        perror("mkdir()");
    }
    
    char newfilename[30], newfilepath[50] = "./";
    printf("Enter file name for the copy in this directory: \n");
    scanf("%s", newfilename);
    
    strcat(newfilepath, dirname);
    strcat(newfilepath, "/");
    strcat(newfilepath, newfilename);

    fd_new = open(newfilepath, O_CREAT | O_WRONLY, S_IRWXU);
    fd_old = open(oldfilename, O_RDONLY);
    
    if (fd_old == -1) {
        perror("open()");
    }
    
    if (fd_new == -1) {
        perror("open()");
    }
    
    if (fstat(fd_old, &fileStat) == -1) {
		perror("fstat()");
	}
    
    //printf("WTF? LET'S SEE %s %ld\n", newfilepath, fileStat.st_size);
    
    if (sendfile(fd_new, fd_old, 0, fileStat.st_size) == -1) {
        perror("sendfile()");
    }

    if (unlink(oldfilename) == -1) {
        perror("unlink()");
    }
    
    if (close(fd_old) == -1) {
        perror("close()");
    }

    if (close(fd_new) == -1) {
        perror("close()");
    }

    fd_new = open(newfilepath, O_RDWR);
    if (read(fd_new, buffer, strlen(str)) == -1) {
        perror("read()");
    }
    
    if (strcmp(buffer, str) == 0) {
        printf("File has been successfully copied to the directory\n");
    } else {
        perror("Something went wrong, file contents changed.\n");
    }
    
    if (close(fd_new) == -1) {
        perror("close()");
    }

    pid_t pid = fork();

    if (pid == 0) {
        fprintf(stdout, "This process is child, pid = ");
        fprintf(stdout, "%ld\n", (size_t)getpid());
        fprintf(stdout, "Parent's pid = ");
        fprintf(stdout, "%ld\n", (size_t)getppid());
        execlp("sl", "", NULL);
    } else if (pid > 0) {
        fprintf(stdout, "This process is parent, pid = ");
        fprintf(stdout, "%ld\n", (size_t)getpid());
        fprintf(stdout, "Parent's pid = ");
        fprintf(stdout, "%ld\n", (size_t)getppid());
        wait(NULL);
    } else if (pid == -1) {
        perror("fork()");
    }
    return 0;
}
