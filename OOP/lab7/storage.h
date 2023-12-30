#ifndef STORAGE_H
#define STORAGE_H

#include "tlist.h"
#include "tbinary_tree.h"
#include "crit.h"


template <class T>
class TStorage
{
private:
	TList<TBinaryTree<T>> storage;

public:
	TStorage() {	
	}

	~TStorage() {
	}

	void Insert(std::shared_ptr<T> item) {
		if (storage.IsEmpty())	//если вообще пустой список, нужно: создать дерево, положить эл-т, положить это дерево в список
		{
			TBinaryTree<T> tree;
			tree.Insert(item);
			storage.Push(tree);
		}
		else
		{
			//берем топчик, смотрим
			TBinaryTree<T> & top = storage.Top();
			if (top.GetCount() < 5)	//если эл-тов в дереве (топ-эл-т списка) < 5
			{
				top.Insert(item);
			}
			else	//если уже 5, то создать дерево, положить эл-т, положить это дерево в список
			{
				TBinaryTree<T> tree;
				tree.Insert(item);
				storage.Push(tree);
			}
		}
		TBinaryTree<T> & top = storage.Top();
		std::cout << "Object was added with index " << storage.GetLength() - 1 << "." << top.GetCount() - 1 << "\n";
	}

	void DeleteByCriteria(IRemoveCriteria<T> &crit) {
		auto it_list = storage.begin();
		while(it_list != storage.end()) {
			TList< std::shared_ptr<T> > figuresToDelete;
			for (auto it_tree = (*it_list)->begin(); it_tree != (*it_list)->end(); it_tree++) {
				if (crit(*it_tree))
					figuresToDelete.Push(*it_tree);
			}
			bool needToDeleteTree = (figuresToDelete.GetLength() == (*it_list)->GetCount());	//надо удалить сразу все дерево
			for (auto it_figlist = figuresToDelete.begin(); it_figlist != figuresToDelete.end(); it_figlist++) {
				(*it_list)->Delete(**it_figlist);
			}
			if (needToDeleteTree){
				auto it_tmp = it_list;
				it_list++;
				storage.Delete(it_tmp);
			}
			else
				it_list++;
		}
	}

	//template <class A>
	friend std::ostream & operator<<(std::ostream & os, TStorage<T>& stor) {
		size_t i = stor.storage.GetLength() -1;
		if (i == -1) {
			std::cout << "================ EMPTY STORAGE ================ ";
		}
		for (auto it_list = stor.storage.begin(); it_list != stor.storage.end(); it_list++) {
			std::cout << "=================== TREE " << i-- << " =================== " << std::endl;
			std::cout << **it_list;
		}
		return os;
	}
};


#endif /* STORAGE_H */
