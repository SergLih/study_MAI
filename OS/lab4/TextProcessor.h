#ifndef TEXTPROCESSOR_H
#define TEXTPROCESSOR_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/file.h>
#include <errno.h>

#define exit_with_error(msg) { perror(msg); exit(1); } 

void handler(const size_t derp);
void menu(const size_t act);
void searchPrefix(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *subString);
void searchSuffix(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *subString);
void searchPart(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *subString);
void checkLimit(size_t * userLimit);
void print(const size_t fd, const size_t fileSize, const size_t RAMLimit, const size_t line);
void replace(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *oldString, char *newString);
void getStats(const size_t fd, const size_t fileSize, const size_t RAMLimit);
size_t checkFileSize(int fd, size_t minFileLimit, size_t maxFileLimit);

#endif

