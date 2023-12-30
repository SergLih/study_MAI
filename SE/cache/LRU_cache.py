from datetime import datetime as dt
import queue

class LRUCache:
    def __init__(self, capacity: int=10) -> None:
        self._capacity   = capacity
        self._hash_table = {}
        self._time_used_resp_q = queue.PriorityQueue()

    def get(self, key: str) -> str:
        if key not in self._hash_table:
            return ''
        self._time_used_resp_q.put((dt.now(), key))
        return self._hash_table[key]

    def set(self, key: str, value: str) -> None:
        if len(self._hash_table) == self._capacity:
            key_to_remove = self._time_used_resp_q.get()[1]
            while key_to_remove not in self._hash_table:
                key_to_remove = self._time_used_resp_q.get()[1]
            del self._hash_table[key_to_remove]

        self._time_used_resp_q.put((dt.now(), key))
        self._hash_table[key] = value

    def rem(self, key: str) -> None:
        if key in self._hash_table.keys():
            del self._hash_table[key]
