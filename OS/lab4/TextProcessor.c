#include "TextProcessor.h"

void searchPart(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *subString)
{
	size_t strSize = strlen(subString);
    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, index = 0, bytepos = 1;
    char *addr;

    while (offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        for (int i = 0; i < size; ++i, ++bytepos) {
            if (addr[i] == subString[index]) {
                ++index;
                for (size_t j = i + 1; (index < strSize) && (j < size); ++index, ++j) {
                    if (subString[index] != addr[j] || isspace(addr[j])) {
                        index = 0;
                        break;
                    }
                }

                if (index == strSize) {
                    printf("First entry was found at %ld byte\n", bytepos);
                    return;
                }
            }
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }
        offset += size;
    }
    printf("Not found\n");
    //return 0;
}

void searchPrefix(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *subString)
{
	size_t strSize = strlen(subString);
    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, index = 0, bytepos = 1;
    char *addr;

    while (offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        int prev = '\n';
        for (int i = 0; i < size; ++i, ++bytepos) {
            if (addr[i] == subString[index] && isspace(prev)) {
                ++index;
                for (size_t j = i + 1; (index < strSize) && (j < size); ++index, ++j) {
                    if (subString[index] != addr[j] || isspace(addr[j])) {
                        index = 0;
                        break;
                    }
                }

                if (index == strSize) {
                    printf("First entry prefix was found at %ld byte\n", bytepos);
                    return;
                }
            }
            prev = addr[i];
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }
        offset += size;
    }
    printf("Not found\n");
    //return 0;
}


void searchSuffix(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *subString) 
{
	size_t strSize = strlen(subString);
    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, index = 0, bytepos = 1;
    char *addr;
    bool found = false;

    while (offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        char next = '\n';

        for (size_t i = 0; i < size; ++i, ++bytepos) {
            index = 0;
            if (addr[i] == subString[index]) {
                ++index;
                for (size_t j = i + 1; (index < strSize) && (j < size); ++index, ++j) {
                    if (j + 1 < size) {
                        next = addr[j + 1];
                    } else {
                        next = '\n';
                    }
                    if (subString[index] != addr[j] || isspace(addr[j])) {
                        index = 0;
                        break;
                    }
                }

                if (index == strSize && isspace(next)) {
                    printf("First entry suffix was found at %ld byte\n", bytepos);
                    found = true;
                    break;
                }
            }
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }
        offset += size;
    }
    if (!found)
        printf("Not found\n");
    //return 0;
}

void checkLimit(size_t *userLimit)
{
    size_t pageSize = getpagesize();
	//printf("%zu %zu %zu\n", pageSize, *userLimit, (*userLimit % pageSize));
    if (*userLimit < pageSize) {
        printf("Mmap limit is less than page size and therefore will be set to %ld bytes\n", pageSize);
        *userLimit = pageSize;
    } else if ((*userLimit % pageSize) != 0) {
        *userLimit = (*userLimit / pageSize + 1) * pageSize; //здесь округляем. ищем ближайшее число, которое больше, но при этом делится на pageSize
        printf("Mmap limit is not aligned with page size (%ld bytes) and therefore will be set to %ld bytes\n", pageSize, *userLimit);
    } 
}

void print(const size_t fd, const size_t fileSize, const size_t RAMLimit, const size_t lineNo)
{
    if (lineNo == 0) { 
        printf("Line number must be positive.\n");
        return;
    }

    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, curLineNo = 1;
    char *addr;
    bool printed = false;

    while (offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        for (size_t i = 0; i < size; ++i) {
            if (curLineNo == lineNo) {
                putchar(addr[i]);
            }
            if (addr[i] == '\n') {
                ++curLineNo;
                if (curLineNo > lineNo) {
                	printed = true;
                	break;
               	}
            }
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap(): ");
        }

        offset += size;
    }
    //printf("%zu, %zu", curLineNo, lineNo);
    if (!printed) {
        printf("Wrong line number. Try again.\n");
    }
}


void getStats(const size_t fd, const size_t fileSize, const size_t RAMLimit)
{
    char *addr;
    size_t offset = 0, lines = 1;
    
    size_t size = (fileSize < RAMLimit) ? fileSize : RAMLimit;

    while(offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        for (size_t i = 0; i < size && addr[i] != '\0'; ++i) {
            if (addr[i] == '\n') {
                ++lines;
            }
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }

        offset += size;
    }

    if (lines > 1) {
        --lines;
    }

    printf("Number of symbols: %zu\nNumber of lines:   %zu\n", fileSize, lines);
}

size_t removeOldString(const size_t fd, const size_t fileSize, const size_t RAMLimit, const size_t from, const size_t to)
{ //предполагаю, что to - не включительная граница.
    if ((from < 0) || (to < 0) || (from > to) || (to - 1 > fileSize)) {
        printf("Incorrect bounds.\n");
        return 0;
    }
    size_t diff = (fileSize == to - 1) ? 1 : fileSize - to;
    char *old = (char *) malloc(sizeof(char) * diff);
    if (old == NULL) {
        exit_with_error("malloc()");
    }
    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, bytepos = 1, k = 0;
    char *addr;

    while (offset < fileSize) {
        addr = (char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        for (size_t i = 0; i < size; ++i, ++bytepos) {
            if (bytepos >= to) {
                old[k] = addr[i];
                ++k;
            }
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }

        offset += size;
    }

    size_t newFileSize = fileSize - to + from;
    if (ftruncate(fd, newFileSize)) {
        exit_with_error("ftruncate()");
    }

    bytepos = 1, offset = 0, k = 0;
    size = (RAMLimit > newFileSize) ? newFileSize : RAMLimit;

    while(offset < newFileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        for (size_t i = 0; i < size; ++i, ++bytepos) {
            if (bytepos >= from) {
                addr[i] = old[k];
                ++k;
            }
        }

        if (msync(addr, size, MS_SYNC) == -1) {
            exit_with_error("msync()");
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }

        offset += size;
    }

    free(old);
    return newFileSize;
}

void insertNewString(const size_t fd, const size_t fileSize, const size_t RAMLimit, char *string, const size_t from, const size_t to)
{
    if ((from < 0) || (to < 0) || (from > to)) {
        printf("Incorrect bounds.\n");
        return;
    }

    char *old = (char*) malloc(sizeof(char) * (fileSize - from));
    if (old == NULL) {
        exit_with_error("Error while inserting new string. malloc()");
    }
    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, bytepos = 1, k = 0;
    char *addr;

    while (offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("Error while inserting new string. mmap()");
        }

        for (size_t i = 0; i < size; ++i, ++bytepos) {
            if (bytepos >= from) {
                old[k] = addr[i];
                ++k;
            }
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("Error while inserting new string. munmap()");
        }

        offset += size;
    }

    size_t newFileSize = fileSize + to - from;
    if (ftruncate(fd, newFileSize)) {
        exit_with_error("Error while inserting new string. ftruncate()");
    }

    size_t l = 0;
    bytepos = 1, offset = 0, k = 0;
    size = (RAMLimit > newFileSize) ? newFileSize : RAMLimit;

    while (offset < newFileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("Error while inserting new string. mmap()");
        }

        for (size_t i = 0; i < size; ++i, ++bytepos) {
            if ((bytepos >= from) && (bytepos < to)) {
                addr[i] = string[l];
                ++l;
            }

            if (bytepos >= to) {
                addr[i] = old[k];
                ++k;
            }
        }

        if (msync(addr, size, MS_SYNC) == -1) {
            exit_with_error("msync()");
        }

        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }

        offset += size;
    }

    free(old);
}

void replace(const size_t fd, const size_t fileSize,  const size_t RAMLimit, char *oldString, char *newString)
{
	size_t sizeOldStr = strlen(oldString);
	size_t sizeNewStr = strlen(newString);
    size_t size = (RAMLimit > fileSize) ? fileSize : RAMLimit;
    size_t offset = 0, index = 0, bytepos = 1, from = 0, to = 0;
    char *addr;
    bool found = false;


    while (offset < fileSize) {
        addr = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            exit_with_error("mmap()");
        }

        for (size_t i = 0; i < size; ++i, ++bytepos) {
            if (addr[i] == oldString[index]) {
                ++index;
                for (size_t j = i + 1; (index < sizeOldStr) && (j < size); ++index, ++j) {
                    if (oldString[index] != addr[j] || addr[j] == '\n') {
                        index = 0;
                        break;
                    }
                }

                if (index == sizeOldStr) {
                    from = bytepos;
                    found = true;
                    index = 0;
                    break;
                }
            }
        }
        if (munmap(addr, size) == -1) {
            exit_with_error("munmap()");
        }
        offset += size;
    }


    if (!found) {
        printf("Not found.\n");
        return;
    }

    size_t newFileSize = removeOldString(fd, fileSize, RAMLimit, from, from + sizeOldStr);
    insertNewString(fd, newFileSize, RAMLimit, newString, from, from + sizeNewStr);
}

size_t checkFileSize(int fd, size_t minFileLimit, size_t maxFileLimit) 
{
	struct stat fileStat;
	if (fd == -1) {
		perror("open()");
		exit(1); 
	}
	if (fstat(fd, &fileStat) == -1) {
		perror("fstat()");
		exit(1);
	}
	if (fileStat.st_size < minFileLimit || fileStat.st_size > maxFileLimit) {
		printf("Error: File size %ld is outside limits [%ld, %ld]. Program will be closed now\n",
		       fileStat.st_size, minFileLimit, maxFileLimit );
		exit(1);
	}
	return fileStat.st_size;
}
