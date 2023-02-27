# Cuckoo HashMap

Cuckoo hashing implementation on C++.

This implementation uses configurable number of hash functions and random walk approach.

## Usage

Copy or download `cuckoo_hashmap.hpp` file to your project and include it.
`HashMap` class will be in the `Cuckoo` namespace.

You can use any `ValueType` with implemented empty constructor, any `KeyType` with implemented `operator==`, which can
be hashed using `std::hash` or a custom `Hash` class.

Run under C++11 standard or newer.

## Tests

`test_hashmap.cpp` file provides unit tests for `Cuckoo::HashMap`.
