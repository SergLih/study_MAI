#ifndef TALLOCATIONBLOCK_H
#define TALLOCATIONBLOCK_H

#include "tlist_alloc.h"

typedef unsigned char Byte;
typedef void * VoidPtr;

template<class T>
class TList;

class TAllocationBlock
{
public:
	TAllocationBlock(size_t size, size_t count);
	void *Allocate();
	void Deallocate(void *ptr);
	bool Empty();
	size_t Size();

	virtual ~TAllocationBlock();

private:
	Byte *_used_blocks;
	TListAlloc<VoidPtr> _free_blocks;
};

#endif /* TALLOCATIONBLOCK_H * */