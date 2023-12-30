#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/io.h>
#include <sys/mman.h>

#include "TextProcessor.h"


void help()
{
	printf("Use '--help' to show this message\n");
	printf("Use '--go' for interactive mode\n\n");
    printf("Use 'FILE RAMLimitInBytes [COMMAND [ARGUMENTS]]'\n\n");
    printf("Possible commands:\n\n");
    printf("--stats\n");
    printf("--print   NumberString\n");
    printf("--search  SubString pr[efix]/s[uffix]/p[art]\n");
    printf("--replace OldSubString [NewSubString]\n");
    exit(0);
}

int main_menu()
{
    int command;
    printf("Avaliable commands:\n");
    printf("0) Select   a file\n");
    printf("1) Print    string\n");
    printf("2) Search   substring\n");
    printf("3) Replace  sudstring\n");
    printf("4) Output   file statistics\n");
	printf("5) Set RAM  limit\n");
	printf("6) Set file size limits\n");
    printf("7) Exit\n");
    scanf("%d", &command);
    return command;
}


int main(int argc, char *argv[])
{
    int command;
	size_t maxRamLimit = 1024;
	size_t minFileLimit = 16;
	size_t maxFileLimit = 16 * 1024 * 1024;

    size_t curFileSize = 0;
    int fd = 0;
    
    if (argc == 1) {
       help();
    } else if (argc == 2) {
        if (strcmp(argv[1], "--help")==0) { // набрана команда help
            help();
        } else {
            char opt[7], strToSearch[256], strToAdd[256], filename[256], oldStr[256], newStr[256];
            filename[0] = '\0';

            maxRamLimit = atoi(argv[1]);
            checkLimit(&maxRamLimit);

            do {
                command = main_menu();
                switch (command) {
                    case 0: {
                        printf("Enter the name of the file: ");
                        scanf("%255s", filename);
                        if (fd > 0)
                            if (close(fd) != 0)
                                exit_with_error("close()");
                        fd = open(filename, O_RDWR);
						curFileSize = checkFileSize(fd, minFileLimit, maxFileLimit);
                        break;
                    }

                    case 1: {
                        printf("Enter the line number to view: ");
                        size_t numStr;
                        int tmp_numStr;
                        scanf("%d", &tmp_numStr);
                        numStr = tmp_numStr > 0 ? tmp_numStr : 0;
                        print(fd, curFileSize, maxRamLimit, numStr);
                        break;
                    }

                    case 2: {
                    	//size_t res; // 0 = not found, numbers start from 1
                        printf("Enter substring to search: ");
                        scanf("%255s", strToSearch);
                        printf("pr[efix]/s[uffix]/p[art]?\n");
                        scanf("%7s", opt);
                        if (strcmp(opt, "prefix") == 0 || strcmp(opt, "pr") == 0) {
                            /*res =*/ searchPrefix(fd, curFileSize, maxRamLimit, strToSearch);
                        } else if (strcmp(opt, "suffix") == 0 || strcmp(opt, "s") == 0) {
                            /*res =*/ searchSuffix(fd, curFileSize, maxRamLimit, strToSearch);
                        } else if (strcmp(opt, "part") == 0 || strcmp(opt, "p") == 0) {
                            /*res =*/ searchPart(fd, curFileSize, maxRamLimit, strToSearch);
                        } else {
                            printf("You have not specified mode: prefix/suffix/part.\n");
                        }
                        /*if(res != 0)
                        	printf("Not found\n");
                        else
                        	printf("First entry was found at %ld byte\n", res);
                    */
                        break;
                    } 

                    case 3: {
                        printf("Enter substring to replace: ");
                        scanf("%255s", oldStr);
                        printf("Enter substring which it will be replaced: ");
                        scanf("%255s", newStr);
                        replace(fd, curFileSize, maxRamLimit, oldStr, newStr);
                        curFileSize = checkFileSize(fd, minFileLimit, maxFileLimit);
                        break;
                    }

                    case 4: {
                        getStats(fd, curFileSize, maxRamLimit);
                        break;
                    }

                    case 5: {
                    	printf("Enter max RAM limit (bytes): ");
                    	scanf("%zu", &maxRamLimit);
                    	checkLimit(&maxRamLimit);
                        break;
                    }
                        
                    case 6: {
                        printf("Enter min FILE limit (bytes): ");
                        scanf("%ld", &minFileLimit);
                        printf("Enter max FILE limit (bytes): ");
                        scanf("%ld", &maxFileLimit);
                        break;
                    }
                    
                    case 7: { 
                    	printf("Now program will be closed\n");
                    	close(fd);
			            break;
			        }
			            
                    default: {
                        printf("Unknown command.\n");
                        break;
                    }
                }
            } while (command != 7);
        }
    } else if (argc > 2) {
       	maxRamLimit = (size_t) atoi(argv[2]);
       	checkLimit(&maxRamLimit);
       	fd = open(argv[1], O_RDWR);
		curFileSize = checkFileSize(fd, minFileLimit, maxFileLimit);
       	
        if (argc == 4) {
		    if (strcmp(argv[3], "--stats") == 0) {
		        getStats(fd, curFileSize, maxRamLimit);
		    } else {
		        printf("Unknown command.\n");
		    }
		} else if (argc == 5) {
		    if (strcmp(argv[3], "--print") == 0) {
		        size_t numStr;
		        int tmp_numStr = atoi(argv[4]);
		        numStr = tmp_numStr > 0 ? tmp_numStr : 0; 
		        print(fd, curFileSize, maxRamLimit, numStr);
		    } else if (strcmp(argv[3], "--replace") == 0) {
		        char empty[0];
		        replace(fd, curFileSize, maxRamLimit, argv[4], empty);
				curFileSize = checkFileSize(fd, minFileLimit, maxFileLimit);
		    } else {
		        printf("Unknown command\n");
		    }
		} else if (argc == 6) {
		    if (strcmp(argv[3], "--search") == 0) {
				size_t res;
		        if (strcmp(argv[5], "part") == 0 || strcmp(argv[5], "p") == 0) {
		            /*res =*/ searchPart(fd, curFileSize, maxRamLimit, argv[4]);
		        } else if (strcmp(argv[5], "prefix") == 0 || strcmp(argv[5], "pr") == 0) {
		            /*res =*/ searchPrefix(fd, curFileSize, maxRamLimit, argv[4]);
		        } else if (strcmp(argv[5], "suffix") == 0 || strcmp(argv[5], "s") == 0) {
		            /*res =*/ searchSuffix(fd, curFileSize, maxRamLimit, argv[4]);
		        } else {
		            printf("You have not specified mode: prefix/suffix/part.\n");;
		        }
		        /*if(res == 0)
                	printf("Not found\n");
                else
                	printf("First entry was found at %ld byte\n", res);
		        */
		    } else if (strcmp(argv[3], "--replace") == 0) {
		        replace(fd, curFileSize, maxRamLimit, argv[4], argv[5]);
				curFileSize = checkFileSize(fd, minFileLimit, maxFileLimit);
		    } else {
		        printf("Unknown command.\n");
		    }

		} else {
			printf("Wrong command and/or arguments.\n\n");
		    help();
		}
		
		close(fd);
    }

    return 0;
}
