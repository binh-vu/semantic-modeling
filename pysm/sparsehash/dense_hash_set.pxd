# distutils: language = c++

from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set

cdef extern from "<sparsehash/dense_hash_set>" namespace "google" nogil:
    cdef cppclass dense_hash_set[T, HashFcn=*, EqualKey=*, Alloc=*]:
        ctypedef T value_type
        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass

        dense_hash_set() except +
        dense_hash_set(dense_hash_set&) except +
        void set_empty_key(const T& key)
        void set_deleted_key(const T& key)

        # Capacity
        bint empty()
        size_t size()
        size_t max_size()

        # Iterators
        iterator begin()
        iterator end()

        # Element lookup
        iterator find(T&)
        size_t count(T&)

        # Modifiers
        void erase(iterator)
        void erase(iterator, iterator)
        size_t erase(T&)
        void clear()
        void insert(T&)