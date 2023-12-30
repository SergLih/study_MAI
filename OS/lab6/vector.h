#ifndef VECTOR_H
#define VECTOR_H

#define MSX 11
#define LSIZE 32

#include "common.h"

typedef struct {
	TId id;
	TMoney debitBalance ;
	TMoney creditBalance ;
	TMoney creditLimit ;
} TAccount;

typedef struct {
 	size_t size;
 	size_t capasity;
 	TAccount* vector;
} TVector;

TVector *VectorCreate (size_t size);
int VectorSearch (TVector *v, TId accountID);
void VectorAdd (TVector *v, TId accountID);
int VectorDestroy (TVector *v);

bool AddMoney(TVector *v, TMoney amount, TId accountID, TAccountType type);
bool WithdrawMoney(TVector *v, TMoney amount, TId accountID, TAccountType type);
bool GetBalance(TVector *v, TId accountID, char* info);

#endif
