#ifndef ASYNC_READY_POOL_CPU_EXECUTOR_H
#define ASYNC_READY_POOL_CPU_EXECUTOR_H
#include <stddef.h>
#include <stdint.h>

/* Capsule name: async_ready_pool.cpu.v1. Keep its Python owner alive during
 * run. Each index [0,count) executes exactly once. run waits for all callbacks
 * before returning, also when a callback reports failure. Return 0 on success.
 * Callbacks must not unwind or recursively run the same executor. */
typedef int (*async_ready_cpu_callback)(void *context, size_t index);
typedef struct {
    uint64_t version;
    size_t workers;
    void *context;
    int (*run)(void *executor, async_ready_cpu_callback callback,
               void *context, size_t count);
} async_ready_cpu_v1;
#endif
