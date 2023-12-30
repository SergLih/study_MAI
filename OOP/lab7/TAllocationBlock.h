#ifndef TALLOCATIONBLOCK_H
#define TALLOCATIONBLOCK_H

#include <iostream>
#include <cstdlib>
#include <memory>
#include "tlist.h"

typedef unsigned char Byte;
typedef void * VoidPtr;

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
	TList<VoidPtr> _free_blocks;
};

#endif /* TALLOCATIONBLOCK_H * */
