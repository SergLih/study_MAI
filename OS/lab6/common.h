#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <zmq.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <unistd.h>
#include <sys/types.h>


typedef enum {
    R_INITIALIZE, R_GET_BALANCE,
	R_DEBIT_ADD,  R_DEBIT_WTHDRAW,  R_DEBIT_SEND,
    R_CREDIT_ADD, R_CREDIT_WTHDRAW, R_CREDIT_SEND
} TRequestType;

typedef enum {A_DEBIT, A_CREDIT} TAccountType;

typedef unsigned int TId;
typedef long long TMoney;

typedef struct {    
  TRequestType action;
  TId          clientId;
  TId          receiverId;
  TMoney       amount;
} TMessage ;

#endif
