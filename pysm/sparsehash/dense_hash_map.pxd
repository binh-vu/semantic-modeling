# distutils: language = c++

from libcpp.pair cimport pair


cdef extern from "<sparsehash/dense_hash_map>" namespace "google" nogil:
    cdef cppclass dense_hash_map[T, U, HashFcn=*, EqualKey=*, Alloc=*]:
        ctypedef T key_type
        ctypedef U data_type
        ctypedef U mapped_type
        ctypedef pair[const T, U] value_type
        cppclass iterator:
            pair[T, U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            pair[T, U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass

        dense_hash_map() except +
        dense_hash_map(dense_hash_map&) except +
        void set_empty_key(const T& key)
        void set_deleted_key(const T& key)

        # Capacity
        bint empty()
        size_t size()
        size_t max_size()

        # Iterators
        iterator begin()
        iterator end()
        const_iterator const_begin "begin"()
        const_iterator const_end "end"()

        # Element access
        U& operator[](T&)

        # Element lookup
        iterator find(T&)
        const_iterator const_find "find"(T&)
        size_t count(T&)

        # Modifiers
        void erase(iterator)
        void erase(iterator, iterator)
        size_t erase(T&)
        void clear()

        void max_load_factor(float)
        float max_load_factor()
