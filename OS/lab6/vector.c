#include "vector.h"

#define DEFAULT_DEBIT 0
#define DEFAULT_CREDIT 0
#define DEFAULT_CREDITLIMIT -5000

TVector *VectorCreate(size_t size) {
    if (size <= 0) {
        perror("Incorrect size!");
        return (NULL);
    }

    TVector *result = (TVector*) malloc(sizeof(TVector));
    if (result == NULL) {
        perror("Error: not enough memory");
        exit(1);
    }

    result->vector = (TAccount*) malloc(sizeof(TAccount) * size);
    if (result->vector == NULL) {
        perror("Error: not enough memory");
        exit(1);
    }

    result->capasity = size;
    result->size = 0;
    return result;
}

void VectorAdd(TVector *v, TId ID) {

    if (v->size == v->capasity) {
        v->capasity *= 2;
        v->vector = (TAccount*) realloc(v->vector, v->capasity * sizeof(TAccount));
        if (v->vector == NULL) {
            perror("Error: not enough memory");
            exit(-1);
        }
    }

    v->vector[v->size].id = ID;
    v->vector[v->size].debitBalance = DEFAULT_DEBIT;
    v->vector[v->size].creditBalance = DEFAULT_CREDIT;
    v->vector[v->size].creditLimit = DEFAULT_CREDITLIMIT;
    v->size++;
}

int VectorSearch(TVector *v, TId ID) {
    for (int i = 0; i < v->size; i++) 
        if (ID == v->vector[i].id) 
            return i;	
    return -1;
}

bool AddMoney(TVector * v, TMoney amount, TId accountID, TAccountType type) {
	if (amount <= 0)
		return false;
	int idx = VectorSearch(v, accountID);
	if (idx < 0)
		return false;

	if (type == A_DEBIT) {
		v->vector[idx].debitBalance += amount;
	} else {
        v->vector[idx].creditBalance += amount; //по умолчанию все деньги переводятся на кредитный счет для погашения долга 
        if (v->vector[idx].creditBalance > 0) { //а излишки переводятся на дебетовый счет  
            v->vector[idx].debitBalance += v->vector[idx].creditBalance;
            v->vector[idx].creditBalance = 0;
        }
    }
    return true;
}

bool WithdrawMoney(TVector * v, TMoney amount, TId accountID, TAccountType type) {		
    if (amount < 0)
        return false;
    int idx = VectorSearch(v, accountID);
    if (idx < 0)
        return false;

    if (type == A_DEBIT) {

        if (v->vector[idx].debitBalance - amount >= 0) {
            v->vector[idx].debitBalance -= amount;
        } else {
            TMoney deb = v->vector[idx].debitBalance;
            TMoney cred = v->vector[idx].creditBalance;
            TMoney limit = v->vector[idx].creditLimit;
            if ((deb + cred - amount) >= limit) { //недостающие средства пробуем снять с кредитного счета
                v->vector[idx].debitBalance = 0;  //если не укаладывается кредитный лимит, то операция не производится
                v->vector[idx].creditBalance -= amount - deb;
            } else 
                return false;
        }
    } else {
        if (v->vector[idx].creditBalance - amount >= v->vector[idx].creditLimit) {
            v->vector[idx].creditBalance -= amount;
        } else
            return false;
    }
    return true;
}

bool GetBalance(TVector * v, TId accountID, char * info) {
    int idx = VectorSearch(v, accountID);
    if (idx < 0)
        return false;
    sprintf(info, "Id: %u\nDebit balance: %lld\nCredit balance: %lld\nCredit limit: %lld",
        v->vector[idx].id, v->vector[idx].debitBalance, v->vector[idx].creditBalance, v->vector[idx].creditLimit);
    return true;
}

int VectorDestroy(TVector *v) {
    free(v->vector);
    free(v);
    return 0;
}
