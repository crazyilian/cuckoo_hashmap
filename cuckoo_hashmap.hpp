// Cuckoo hash map with multiple hash functions using the random walk approach.

// Created by Andrianov Ilian in January 2023

// References:
// https://www.cs.toronto.edu/~noahfleming/CuckooHashing.pdf - Cuckoo Hashing
// https://www.math.cmu.edu/~af1p/Texfiles/cuckoo.pdf - Cuckoo Hashing with random walk
// https://cs.stanford.edu/~rishig/courses/ref/l13a.pdf - Cuckoo Hashing performance analysis

#pragma once

#include <random>
#include <list>
#include <array>
#include <vector>
#include <chrono>
#include <stdexcept>

namespace Cuckoo {

namespace {
constexpr double MAX_LOAD_FACTOR = 0.8;
constexpr size_t MAX_EVICT_LOOP_ITERATIONS = 6;
constexpr size_t HASH_NUMBER = 3;
}

template<class KeyType, class ValueType, class Hash=std::hash<KeyType>>
class HashMap {
private:
  using KeyValueType = std::pair<const KeyType, ValueType>;
  using hash_t = size_t;  // NOLINT

  struct Node {
    std::list<KeyValueType> key_vals;
    hash_t key_hash;
    Node *next;

    Node();
    Node(const KeyValueType &key_val, hash_t key_hash, Node *next);
    Node(const std::list<KeyValueType> &key_vals, hash_t key_hash, Node *next);
    void delete_empty_next();
  };

  using DataType = std::array<std::vector<Node *>, HASH_NUMBER>;

  struct CuckooHasher {
    hash_t A, B, C; // NOLINT
    uint8_t log; // NOLINT

    CuckooHasher();
    template<class RandomFunction>
    CuckooHasher(const uint8_t log, RandomFunction &rnd);
    size_t operator()(const hash_t x) const;
  };

  uint8_t capacity_log_;
  size_t capacity_;
  size_t size_;
  Hash key_hasher_;
  std::mt19937_64 rnd_;
  std::array<CuckooHasher, HASH_NUMBER> hashers_;
  DataType data_;
  Node *begin_ptr_, *end_ptr_;

public:
  class iterator {  // NOLINT
  private:
    using KeyValueIterator = typename std::list<KeyValueType>::iterator;

    Node *ptr_;
    KeyValueIterator it_;

  public:
    iterator();
    explicit iterator(Node *node_ptr, const KeyValueIterator &key_val_it);

    bool operator==(const iterator &other) const;
    bool operator!=(const iterator &other) const;

    KeyValueType &operator*() const;
    KeyValueType *operator->() const;

    iterator &operator++();
    iterator operator++(int);

    friend class HashMap;
  };

  class const_iterator {  // NOLINT
  private:
    using KeyValueIterator = typename std::list<KeyValueType>::const_iterator;

    Node *ptr_;
    KeyValueIterator it_;

  public:
    const_iterator();
    const_iterator(const iterator &it);
    explicit const_iterator(Node *node_ptr, const KeyValueIterator &key_val_it);

    bool operator==(const const_iterator &other) const;
    bool operator!=(const const_iterator &other) const;

    const KeyValueType &operator*() const;
    const KeyValueType *operator->() const;

    const_iterator &operator++();
    const_iterator operator++(int);

    friend class HashMap;
  };

private:
  bool is_overload() const;

  void delete_empty_begin();
  void delete_all_pointers();

  void rehash();
  bool push_to_data(Node *cur_ptr, DataType &data);
  iterator find_by_hash(const KeyType &key, hash_t key_hash) const;
  KeyValueType &insert_new_node(const KeyValueType &key_val, hash_t key_hash);

public:
  explicit HashMap(const Hash &key_hasher = Hash());
  template<class InputIterator>
  HashMap(const InputIterator &begin, const InputIterator &end, const Hash &key_hasher = Hash());
  HashMap(const std::initializer_list<KeyValueType> &lst, const Hash &key_hasher = Hash());
  HashMap(const HashMap &other);
  ~HashMap();
  HashMap &operator=(const HashMap &other);

  size_t size() const;
  bool empty() const;
  Hash hash_function() const;

  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;

  void insert(const KeyValueType &key_val);
  void erase(const KeyType &key);
  void clear();

  iterator find(const KeyType &key);
  const_iterator find(const KeyType &key) const;
  ValueType &operator[](const KeyType &key);
  const ValueType &at(const KeyType &key) const;
};

/**********************************************************************************************************************
 *                                                   HashMap::Node                                                    *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::Node::Node() : next(nullptr) {}

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::Node::Node(const KeyValueType &key_val, hash_t key_hash, Node *next)
    : key_vals({key_val,}), key_hash(key_hash), next(next) {}

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::Node::Node(const std::list<KeyValueType> &key_vals, hash_t key_hash, Node *next)
    : key_vals(key_vals), key_hash(key_hash), next(next) {}

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::Node::delete_empty_next() {
  /**
   * Deleting all next empty nodes (except for fake last node).
   * So iterating is amortized O(n).
   */
  if (next == nullptr) {
    return;
  }
  while (next->next != nullptr && next->key_vals.
      empty()
      ) {
    auto new_next = next->next;
    delete
        next;
    next = new_next;
  }
}

/**********************************************************************************************************************
 *                                               HashMap::CuckooHasher                                                *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::CuckooHasher::CuckooHasher() = default;

template<class KeyType, class ValueType, class Hash>
template<class RandomFunction>
HashMap<KeyType, ValueType, Hash>::CuckooHasher::CuckooHasher(const uint8_t log, RandomFunction &rnd)
    : A(rnd()), B(rnd()), C(rnd()), log(log) {}

template<class KeyType, class ValueType, class Hash>
size_t HashMap<KeyType, ValueType, Hash>::CuckooHasher::operator()(const hash_t x) const {
  return ((A * x) ^ (B * x) ^ (C * x)) >> (sizeof(hash_t) * 8 - log);
}

/**********************************************************************************************************************
 *                                                 HashMap::iterator                                                  *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::iterator::iterator() = default;

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::iterator::iterator(Node *node_ptr, const KeyValueIterator &key_val_it)
    : ptr_(node_ptr), it_(key_val_it) {}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::iterator::operator==(const iterator &other) const {
  return it_ == other.it_;
}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::iterator::operator!=(const iterator &other) const {
  return it_ != other.it_;
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::KeyValueType &HashMap<KeyType,
                                                                  ValueType,
                                                                  Hash>::iterator::operator*() const {
  return *it_;
}
template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::KeyValueType *HashMap<KeyType,
                                                                  ValueType,
                                                                  Hash>::iterator::operator->() const {
  return &(*it_);
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::iterator &HashMap<KeyType, ValueType, Hash>::iterator::operator++() {
  ++it_;
  if (it_ != ptr_->key_vals.end()) {
    // Increased key-value iterator is correct
    return *this;
  }
  // Increased key-value incorrect (no more key-values in this node).
  // Deleting all next empty nodes and going to new node.
  ptr_->delete_empty_next();
  ptr_ = ptr_->next;
  it_ = ptr_->key_vals.begin();
  return *this;
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::iterator HashMap<KeyType, ValueType, Hash>::iterator::operator++(int) {
  auto copy = *this;
  ++(*this);
  return copy;
}

/**********************************************************************************************************************
 *                                              HashMap::const_iterator                                               *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::const_iterator::const_iterator() = default;

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::const_iterator::const_iterator(const iterator &it) : ptr_(it.ptr_), it_(it.it_) {}

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::const_iterator::const_iterator(Node *node_ptr, const KeyValueIterator &key_val_it)
    : ptr_(node_ptr), it_(key_val_it) {}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::const_iterator::operator==(const const_iterator &other) const {
  return it_ == other.it_;
}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::const_iterator::operator!=(const const_iterator &other) const {
  return it_ != other.it_;
}

template<class KeyType, class ValueType, class Hash>
const typename HashMap<KeyType, ValueType, Hash>::KeyValueType &HashMap<KeyType,
                                                                        ValueType,
                                                                        Hash>::const_iterator::operator*() const {
  return *it_;
}
template<class KeyType, class ValueType, class Hash>
const typename HashMap<KeyType, ValueType, Hash>::KeyValueType *HashMap<KeyType,
                                                                        ValueType,
                                                                        Hash>::const_iterator::operator->() const {
  return &(*it_);
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::const_iterator &HashMap<KeyType,
                                                                    ValueType,
                                                                    Hash>::const_iterator::operator++() {
  ++it_;
  if (it_ != ptr_->key_vals.end()) {
    // Increased key-value iterator is correct
    return *this;
  }
  // Increased key-value incorrect (no more key-values in this node).
  // Deleting all next empty nodes and going to new node.
  ptr_->delete_empty_next();
  ptr_ = ptr_->next;
  it_ = ptr_->key_vals.begin();
  return *this;
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::const_iterator HashMap<KeyType,
                                                                   ValueType,
                                                                   Hash>::const_iterator::operator++(int) {
  auto copy = *this;
  ++(*this);
  return copy;
}

/**********************************************************************************************************************
 *                               HashMap constructors, destructor, assignment operators                               *
 **********************************************************************************************************************/


template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::HashMap(const Hash &key_hasher)
    : capacity_log_(6), // Initial capacity is 2**6 (no small rehashes)
      size_(0),
      key_hasher_(key_hasher),
      rnd_(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
  begin_ptr_ = end_ptr_ = new Node();
  rehash();
}

template<class KeyType, class ValueType, class Hash>
template<class InputIterator>
HashMap<KeyType, ValueType, Hash>::HashMap(const InputIterator &begin,
                                           const InputIterator &end,
                                           const Hash &key_hasher) : HashMap(key_hasher) {
  for (auto it = begin; it != end; ++it) {
    insert(*it);
  }
}

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::HashMap(const std::initializer_list<KeyValueType> &lst,
                                           const Hash &key_hasher) : HashMap(key_hasher) {
  for (const auto &key_val : lst) {
    insert(key_val);
  }
}
template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::HashMap(const HashMap &other) {
  begin_ptr_ = end_ptr_ = nullptr;
  *this = other;
}

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash>::~HashMap() {
  delete_all_pointers();
}

template<class KeyType, class ValueType, class Hash>
HashMap<KeyType, ValueType, Hash> &HashMap<KeyType,
                                           ValueType,
                                           Hash>::HashMap::operator=(const HashMap &other) {
  if (this == &other) {
    return *this;
  }
  delete_all_pointers();
  capacity_log_ = other.capacity_log_;
  capacity_ = other.capacity_;
  size_ = other.size_;
  key_hasher_ = other.key_hasher_;
  rnd_ = other.rnd_;
  hashers_ = other.hashers_;
  begin_ptr_ = end_ptr_ = new Node();
  // Iterate all data nodes. Create and add nodes to node list.
  for (size_t hash_ind = 0; hash_ind < HASH_NUMBER; ++hash_ind) {
    data_[hash_ind].resize(capacity_);
    for (size_t h = 0; h < capacity_; ++h) {
      const auto &node_ptr = other.data_[hash_ind][h];
      if (node_ptr != nullptr) {
        begin_ptr_ = data_[hash_ind][h] = new Node(node_ptr->key_vals, node_ptr->key_hash, begin_ptr_);
      } else {
        data_[hash_ind][h] = nullptr;
      }
    }
  }
  return *this;
}

/**********************************************************************************************************************
 *                                                  HashMap getters                                                   *
 **********************************************************************************************************************/


template<class KeyType, class ValueType, class Hash>
size_t HashMap<KeyType, ValueType, Hash>::size() const {
  return size_;
}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::empty() const {
  return size_ == 0;
}

template<class KeyType, class ValueType, class Hash>
Hash HashMap<KeyType, ValueType, Hash>::hash_function() const {
  return key_hasher_;
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::iterator HashMap<KeyType, ValueType, Hash>::begin() {
  return iterator(begin_ptr_, begin_ptr_->key_vals.begin());
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::const_iterator HashMap<KeyType, ValueType, Hash>::begin() const {
  return const_iterator(begin_ptr_, begin_ptr_->key_vals.begin());
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::iterator HashMap<KeyType, ValueType, Hash>::end() {
  return iterator(end_ptr_, end_ptr_->key_vals.end());
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::const_iterator HashMap<KeyType, ValueType, Hash>::end() const {
  return const_iterator(end_ptr_, end_ptr_->key_vals.end());
}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::is_overload() const {
  /**
   * Check if load factor exceeds the load factor limit.
   */
  return size_ >= MAX_LOAD_FACTOR * capacity_;
}

/*********************************************************************************************************************
 *                                                 HashMap modifiers                                                 *
 *********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::insert(const KeyValueType &key_val) {
  /**
   * Insert new key-value pair to hash map
   */
  hash_t key_hash = key_hasher_(key_val.first);
  const auto &it = find_by_hash(key_val.first, key_hash);
  const auto &key_hash_node = it.ptr_;
  const auto &key_it = it.it_;
  if (key_hash_node == end_ptr_) {
    // No node with right key_hash
    insert_new_node(key_val, key_hash);
    return;
  }
  auto &key_vals = key_hash_node->key_vals;
  if (key_it == key_vals.end()) {
    // Key is new
    ++size_;
    key_vals.push_back(key_val);
  }
}

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::erase(const KeyType &key) {
  /**
   * Erase key from hash map
   */
  auto key_hash = key_hasher_(key);
  for (size_t hash_ind = 0; hash_ind < HASH_NUMBER; ++hash_ind) {
    auto h = hashers_[hash_ind](key_hash);
    auto &node_ptr = data_[hash_ind][h];
    if (node_ptr != nullptr && node_ptr->key_hash == key_hash) {
      // Found node with right key_hash
      auto &key_vals = node_ptr->key_vals;
      for (auto key_val_it = key_vals.begin(); key_val_it != key_vals.end(); ++key_val_it) {
        if (key_val_it->first == key) {
          // Found key we are looking for
          --size_;
          key_vals.erase(key_val_it);
          if (key_vals.empty()) {
            node_ptr = nullptr;
            delete_empty_begin();
          }
          return;
        }
      }
    }
  }
}

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::clear() {
  /**
   * Clear all hash map
   */
  size_ = 0;
  for (size_t hash_ind = 0; hash_ind < HASH_NUMBER; ++hash_ind) {
    for (auto &node_ptr : data_[hash_ind]) {
      node_ptr = nullptr;
    }
  }
  delete_all_pointers();
  begin_ptr_ = end_ptr_ = new Node();
}

/**********************************************************************************************************************
 *                                            HashMap key-value accessors                                             *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::iterator HashMap<KeyType, ValueType, Hash>::find(const KeyType &key) {
  /**
   * Search for key-value pair by key
   */
  auto it = find_by_hash(key, key_hasher_(key));
  if (it.ptr_->key_vals.end() == it.it_) {
    return end();
  }
  return it;
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::const_iterator HashMap<KeyType,
                                                                   ValueType,
                                                                   Hash>::find(const KeyType &key) const {
  /**
   * Search for key-value pair by key
   */
  auto it = find_by_hash(key, key_hasher_(key));
  if (it.ptr_->key_vals.end() == it.it_) {
    return end();
  }
  return it;
}

template<class KeyType, class ValueType, class Hash>
ValueType &HashMap<KeyType, ValueType, Hash>::operator[](const KeyType &key) {
  /**
   * Return a reference to value by key. Insert key with default value if key doesn't exists.
   */
  hash_t key_hash = key_hasher_(key);
  auto it = find_by_hash(key, key_hash);
  if (it.ptr_ == end_ptr_) {
    // No node with right key_hash
    return insert_new_node({key, ValueType()}, key_hash).second;
  }
  auto &key_vals = it.ptr_->key_vals;
  if (it.it_ == key_vals.end()) {
    // No right key in right node
    ++size_;
    key_vals.push_back({key, ValueType()});
    return key_vals.back().second;
  }
  // Key-value pair found
  return it->second;
}

template<class KeyType, class ValueType, class Hash>
const ValueType &HashMap<KeyType, ValueType, Hash>::at(const KeyType &key) const {
  /**
   * Return a reference to value by key. Throw exception if key doesn't exists.
   */
  hash_t key_hash = key_hasher_(key);
  auto it = find_by_hash(key, key_hash);
  if (it.ptr_->key_vals.end() == it.it_) {
    throw std::out_of_range("Key doesn't exists");
  }
  return it->second;
}

/**********************************************************************************************************************
 *                                              HashMap memory cleaners                                               *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::delete_empty_begin() {
  /**
   * Delete all empty nodes in front of the node list.
   */
  while (begin_ptr_ != end_ptr_ && begin_ptr_->key_vals.empty()) {
    auto next = begin_ptr_->next;
    delete begin_ptr_;
    begin_ptr_ = next;
  }
}

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::delete_all_pointers() {
  /**
   * Delete all nodes in node list.
   */
  auto ptr = begin_ptr_;
  while (ptr != nullptr) {
    auto next = ptr->next;
    delete ptr;
    ptr = next;
  }
  begin_ptr_ = end_ptr_ = nullptr;
}

/**********************************************************************************************************************
 *                                               HashMap help functions                                               *
 **********************************************************************************************************************/

template<class KeyType, class ValueType, class Hash>
void HashMap<KeyType, ValueType, Hash>::rehash() {
  /**
   * Create new hashers and insert all key-value pairs according to the new rules.
   * If some key can't be inserted, then rehash again.
   */

  // Updating capacity if changed
  capacity_ = (1 << capacity_log_) + 1;
  while (true) {
    // Create new hashers
    for (auto &hasher : hashers_) {
      hasher = CuckooHasher(capacity_log_, rnd_);
    }
    // Creating empty new data array
    DataType new_data;
    for (size_t hash_ind = 0; hash_ind < HASH_NUMBER; ++hash_ind) {
      new_data[hash_ind].resize(capacity_, nullptr);
    }
    // Rehash all nodes
    bool successful = true;
    for (auto it = begin(); it != end(); ++it) {
      if (!push_to_data(it.ptr_, new_data)) {
        successful = false;
        break;
      }
    }
    if (successful) {
      // Saving rehashed data
      std::swap(data_, new_data);
      return;
    }
  }
}

template<class KeyType, class ValueType, class Hash>
bool HashMap<KeyType, ValueType, Hash>::push_to_data(Node *cur_ptr, DataType &data) {
  /**
   * Try to insert a key with random walk approach.
   * No rehashing if loop iteration count exceeded.
   * Returns boolean - was insertion successful (true) or some key was evicted and not inserted (false).
   */

  // No restricted hashers in first iteration
  size_t restricted_hash_ind = HASH_NUMBER;
  bool is_restricted = false;

  for (size_t loop_iteration = 0; loop_iteration < MAX_EVICT_LOOP_ITERATIONS; ++loop_iteration) {
    // Evict random node and place current node there
    {
      size_t hash_ind = rnd_() % (HASH_NUMBER - is_restricted);
      if (hash_ind >= restricted_hash_ind) {
        ++hash_ind;
      }
      auto h = hashers_[hash_ind](cur_ptr->key_hash);
      auto &old_ptr = data[hash_ind][h];
      if (old_ptr == nullptr) {
        // Free space. Insert current node and finish pushing
        old_ptr = cur_ptr;
        return true;
      } else if (old_ptr->key_hash == cur_ptr->key_hash) {
        // Copy current key-values to old key-values node because they have equal key hashes
        auto &old_key_vals = old_ptr->key_vals;
        old_key_vals.splice(old_key_vals.end(), cur_ptr->key_vals);
        return true;
      }
      // Placing current node by evicting old node
      std::swap(cur_ptr, old_ptr);
      // Prevent insertion of evicted node to old location by restricting this hash index at next iteration
      is_restricted = true;
      restricted_hash_ind = hash_ind;
    }
    // Try to insert evicted node if there is free space
    for (size_t hash_ind = 0; hash_ind < HASH_NUMBER; ++hash_ind) {
      auto h = hashers_[hash_ind](cur_ptr->key_hash);
      auto &node = data[hash_ind][h];
      if (node == nullptr) {
        node = cur_ptr;
        return true;
      }
    }
    // Try again with new evicted key
  }
  return false;
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::iterator HashMap<KeyType,
                                                             ValueType,
                                                             Hash>::find_by_hash(const KeyType &key,
                                                                                 hash_t key_hash) const {
  /**
   * Returns a pair of node pointer and iterator on key-value pair in this node.
   * If no node with right key_hash exists returns pointer to fake last empty node.
   * If node with right key_hash exists but there is no right key, it returns right pointer and iterator to
   * end of key-value list.
   */

  // Iterate each hash (node can be in any of them)
  for (size_t hash_ind = 0; hash_ind < HASH_NUMBER; ++hash_ind) {
    auto h = hashers_[hash_ind](key_hash);
    auto &node_ptr = data_[hash_ind][h];
    if (node_ptr != nullptr && node_ptr->key_hash == key_hash) {
      // Found node with right key_hash. If key exists, is it only here
      auto &key_vals = node_ptr->key_vals;
      for (auto key_val_it = key_vals.begin(); key_val_it != key_vals.end(); ++key_val_it) {
        if (key_val_it->first == key) {
          return iterator(node_ptr, key_val_it);
        }
      }
      // No right key found
      return iterator(node_ptr, key_vals.end());
    }
  }
  // No right node found
  return iterator(end_ptr_, end_ptr_->key_vals.end());
}

template<class KeyType, class ValueType, class Hash>
typename HashMap<KeyType, ValueType, Hash>::KeyValueType &HashMap<KeyType,
                                                                  ValueType,
                                                                  Hash>::insert_new_node(const KeyValueType &key_val,
                                                                                         hash_t key_hash) {
  /**
   * Insert key-value considering there is no node with such key hash.
   * Return link to inserted key-value pair.
   */
  // Create new node and add to node list (to front).
  Node *node = new Node(key_val, key_hash, begin_ptr_);
  begin_ptr_ = node;
  ++size_;
  if (is_overload()) {
    // Increase capacity: multiply by 4 if capacity < 2**18 ≈ 2.6e5; multiply by 2 times in other case.
    ++capacity_log_;
    if (capacity_log_ <= 18) {
      ++capacity_log_;
    }
    // New node will be inserted during rehashing, because it is already in the node list.
    rehash();
  } else if (!push_to_data(node, data_)) {
    // Try to insert new node to data. If that fails, just rehash
    // New node will be inserted during rehashing, because it is already in the node list.
    rehash();
  }
  return node->key_vals.front();
}

}