from LRU_cache import LRUCache

cache = LRUCache(100)
cache.set('Jesse', 'Pinkman')
cache.set('Walter', 'White')
cache.set('Jesse', 'James')
assert cache.get('Jesse') == 'James'  # вернёт 'James'
#print(cache.get('Jesse'))
cache.rem('Walter')
#print(cache.get('Walter'))
assert cache.get('Walter') == '' # вернёт ''
