#pragma once

// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
// hood
// User ptr is used if and only if it is properly aligned (page aligned). When
// not
// aligned, runtime has no choice but to create its own host side buffer that
// backs
// user ptr. This in turn implies that all operations that move data to and from
// device incur an extra memcpy to move data to/from runtime's own host buffer
// from/to user pointer. So it is recommended to use this allocator if user wish
// to
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to
// the
// page boundary. It will ensure that user buffer will be used when user create
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.
template <typename T>
struct aligned_allocator {

  using value_type = T;

  T *allocate(std::size_t num) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t num) {
	  free(p);
  }
};
