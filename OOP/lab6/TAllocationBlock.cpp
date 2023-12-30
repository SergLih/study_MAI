#include "TAllocationBlock.h"
#include <iostream>

TAllocationBlock::TAllocationBlock(size_t size, size_t count)
{
	_used_blocks = (Byte *)malloc(size * count);

  void * ptr;
	for (size_t i = 0; i < count; ++i) {
		ptr = _used_blocks + i * size;
		_free_blocks.Push(ptr);
	}
}

void *TAllocationBlock::Allocate()
{
	if (!_free_blocks.IsEmpty()) {
		void * res = _free_blocks.Top();
		_free_blocks.Pop();
		return res;
	}
	else {
		throw std::bad_alloc();
	}
}

void TAllocationBlock::Deallocate(void *ptr)
{
	_free_blocks.Push(ptr);
}

bool TAllocationBlock::Empty()
{
	return _free_blocks.IsEmpty();
}

size_t TAllocationBlock::Size()
{
	return _free_blocks.GetLength();
}

TAllocationBlock::~TAllocationBlock()
{
	free(_used_blocks);
	std::cout << "allocator finished it\'s work\n"; 
}
