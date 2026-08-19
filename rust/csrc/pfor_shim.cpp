#include <cstdint>
#include <cstddef>
#include <memory>
#include "codecfactory.h"
using namespace FastPForLib;

static IntegerCODEC& codec256() {
    // Keep the factory alive as a thread_local: getFromName returns a reference into the
    // factory's map, so the factory must outlive the shared_ptr copy below. (A temporary
    // factory would also work here since the by-value shared_ptr copies ownership during
    // the full-expression, but a named thread_local makes the lifetime obvious.)
    static thread_local CODECFactory factory;
    static thread_local std::shared_ptr<IntegerCODEC> c = factory.getFromName("simdfastpfor256");
    return *c;
}

extern "C" size_t fastpfor256_decode(const uint32_t* in, size_t in_len,
                                     uint32_t* out, size_t n) {
    size_t nvalue = n;                 // in: expected count; out: decoded count
    try {
        codec256().decodeArray(in, in_len, out, nvalue);
        return nvalue;
    } catch (...) {
        return 0;
    }
}

extern "C" size_t fastpfor256_encode(const uint32_t* in, size_t n,
                                     uint32_t* out, size_t out_cap) {
    // Caller (ffi_encode) never passes n==0; codec256().encodeArray sets nvalue to the
    // number of u32 words written (== pyfastpfor `wl`). Return SIZE_MAX on any throw so
    // the Rust side can surface a clean error instead of trusting garbage output.
    size_t nvalue = out_cap;
    try {
        codec256().encodeArray(in, n, out, nvalue);
        return nvalue;
    } catch (...) {
        return static_cast<size_t>(-1);
    }
}
