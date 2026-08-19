// #269 -- ASan/UBSan gate over the vendored FastPFor decode path.
//
// WHY THIS EXISTS, AND WHY AN EXIT-CODE TEST IS NOT ENOUGH
// -------------------------------------------------------
// The pytest regression suite can only observe an out-of-bounds access that leaves mapped
// memory and kills the process. Defect B is a *read*, and a read that lands inside the heap
// is completely invisible to it. That is not hypothetical here: pre-fix, the `cexcept-255`
// mutation decoded out of bounds, returned the expected value COUNT, and was therefore
// accepted -- silent data corruption with no crash and no error. Only a sanitizer sees it.
//
// Rust nightly is not installed on the development host, so `-Zsanitizer=address` over the
// whole extension is unavailable. This is the documented fallback: build ONLY the C++ shim
// plus the vendored FastPFor sources with `-fsanitize=address,undefined` and drive
// `fastpfor256_decode` directly. See `run.sh` for the build/run recipe and for the
// negative-control build that proves this gate can actually detect the defect.
//
// Output buffers are sized the way the FIXED Rust/Python callers now size them
// (expected + input_bytes), so any sanitizer report here is an INPUT-side violation --
// exactly what defect B is about. Every buffer is a fresh exact-size heap allocation, so a
// read or write one element past either end is a heap-buffer-overflow.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" size_t fastpfor256_decode(const uint32_t *in, size_t in_len, uint32_t *out, size_t n);
extern "C" size_t fastpfor256_encode(const uint32_t *in, size_t n, uint32_t *out, size_t out_cap);

// Exit statuses. 1 is reserved for the sanitizer (it aborts the process itself), so a harness
// fault must never look like a sanitizer report -- run.sh distinguishes them.
enum { EXIT_HARNESS_WRONG_RESULT = 2, EXIT_HARNESS_FAULT = 3 };

static long g_decodes = 0;
static long g_rejected = 0;
static long g_accepted = 0;

// Never hand the decoder a null pointer: malloc(0) is permitted to return NULL, the truncation
// sweep below decodes a zero-word input, and decodeArray forms `initin + length` before it
// checks `length == 0`. A null there would be a HARNESS bug reported as a FastPFor one.
static void *xmalloc(size_t nbytes) {
  void *p = malloc(nbytes ? nbytes : 1);
  if (p == nullptr) {
    fprintf(stderr, "FATAL: harness allocation of %zu bytes failed\n", nbytes);
    exit(EXIT_HARNESS_FAULT);
  }
  return p;
}

// Decode with an exact-size heap input copy and the fixed caller's output allocation.
static size_t try_decode(const uint32_t *words, size_t nwords, size_t n) {
  uint32_t *in = static_cast<uint32_t *>(xmalloc(nwords * sizeof(uint32_t)));
  if (nwords) memcpy(in, words, nwords * sizeof(uint32_t));
  size_t outcap = n + nwords * 4;  // == expected + input_bytes, the fixed bound
  uint32_t *out = static_cast<uint32_t *>(xmalloc(outcap * sizeof(uint32_t)));
  size_t got = fastpfor256_decode(in, nwords, out, n);
  ++g_decodes;
  if (got == n) ++g_accepted; else ++g_rejected;
  free(in);
  free(out);
  return got;
}

static std::vector<uint32_t> encode(const std::vector<uint32_t> &src) {
  std::vector<uint32_t> out(src.size() + 4096);
  size_t wl = fastpfor256_encode(src.data(), src.size(), out.data(), out.size());
  if (wl == static_cast<size_t>(-1) || wl > out.size()) {
    fprintf(stderr, "FATAL: encode failed for %zu values\n", src.size());
    exit(EXIT_HARNESS_WRONG_RESULT);
  }
  out.resize(wl);
  return out;
}

// ---- the PRECISE negative control -------------------------------------------------------
//
// Over-claim a block's exception count. This is the one mutation in
// tests/cell/test_pfor_rust_kernel.py that pre-fix neither crashed nor raised: the walk keeps
// reading exception-position bytes past the end of the exception byte container, and its first
// invalid read lands one byte past the end of the input allocation. That byte is mapped, so the
// process survives, `got == expected` is satisfied, and the corrupt frame is ACCEPTED. Silent
// data corruption, invisible to every exit-code test in the suite. (ASan halts at that first
// invalid byte; unsanitized, the walk simply carries on.)
//
// It is isolated behind `gate cexcept-overclaim` because the full sweep below is NOT a valid
// negative control on its own: pre-fix it dies on the very first wheremeta case, a wild pointer
// that would segfault with or without a sanitizer, and so proves nothing about reads that stay
// inside mapped memory. Measured pre-fix (20768f3d), this case reports:
//     AddressSanitizer: heap-buffer-overflow ... READ of size 1
//     0 bytes to the right of 356-byte region
//     #0 FastPForLib::SIMDFastPFor<8u>::__decodeArray
// and on the current tree it is rejected cleanly (got=0) with no report.
static int cexcept_overclaim() {
  // Same shape as the Python fixture `_valid_block_frame`: exactly one 256-value block with
  // real exceptions.
  std::vector<uint32_t> src(256);
  std::mt19937 rng(4);
  for (auto &v : src) v = rng() % 900u;
  src[10] = 0xFFFFFFFFu;
  src[77] = 5000u;
  std::vector<uint32_t> blob = encode(src);

  const uint32_t wheremeta = blob[1];
  const size_t bc = (static_cast<size_t>(wheremeta) + 2) * 4;  // byte offset of [b, cexcept, maxbits]
  if (bc + 3 > blob.size() * 4) {
    fprintf(stderr, "FATAL: per-block bytes at %zu outside a %zu-word frame\n", bc, blob.size());
    return EXIT_HARNESS_WRONG_RESULT;
  }
  uint8_t *raw = reinterpret_cast<uint8_t *>(blob.data());
  const uint8_t b = raw[bc], cexcept = raw[bc + 1], maxbits = raw[bc + 2];
  printf("frame: %zu words, count=%u wheremeta=%u | b=%u cexcept=%u maxbits=%u\n",
         blob.size(), blob[0], wheremeta, b, cexcept, maxbits);
  // Preconditions, mirroring the Python fixture's asserts. Without exceptions there is no
  // exception-position walk to run off the end, and cexcept must have room to be over-claimed.
  if (blob[0] != 256 || cexcept == 0 || cexcept >= 255 || b >= maxbits || maxbits > 32) {
    fprintf(stderr, "FATAL: fixture preconditions not met -- this case tests nothing\n");
    return EXIT_HARNESS_WRONG_RESULT;
  }
  if (try_decode(blob.data(), blob.size(), 256) != 256) {
    fprintf(stderr, "FATAL: the UNMUTATED frame did not decode\n");
    return EXIT_HARNESS_WRONG_RESULT;
  }

  raw[bc + 1] = 255;
  // Flushed marker, checked by run.sh. Without it a sanitizer report raised by the UNMUTATED
  // decode above would look identical to one raised by the mutation, and the control would
  // pass while testing the wrong thing. The flush is required, not tidiness: the sanitizer
  // aborts the process, and stdout is fully buffered when run.sh captures it through a pipe.
  printf("MUTATED DECODE REACHED\n");
  fflush(stdout);
  const size_t got = try_decode(blob.data(), blob.size(), 256);
  printf("cexcept=255: got=%zu (%s)\n", got,
         got == 256 ? "ACCEPTED -- corrupt frame decoded as valid" : "rejected");
  printf("NEGATIVE CONTROL CASE COMPLETE -- no sanitizer report\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1) {
    if (strcmp(argv[1], "cexcept-overclaim") == 0) return cexcept_overclaim();
    fprintf(stderr, "unknown case: %s\n", argv[1]);
    return EXIT_HARNESS_FAULT;
  }

  // ---- 1. the explicit malformed inputs from the regression tests -------------------
  // The stage-1 count MUST be a nonzero multiple of BlockSize (256), or __decodeArray never
  // examines a single displacement and the vector tests nothing. Two different mechanisms, and
  // only one of them is an error: a non-multiple hits the divisibility throw, whereas ZERO is
  // perfectly legal -- it is what every frame of fewer than 256 values carries, decoded entirely
  // by the VariableByte tail -- and merely leaves the page loop with nothing to do. A poisoned
  // zero-count vector is therefore rejected only by the closing `got != expected` check, on
  // whatever the tail happened to decode from the remaining bytes. The checkpoint-2 review
  // caught exactly this in the first draft, where every vector started with a count of 1.
  {
    const uint32_t wm[] = {256, 0x20000000u, 0, 0, 0, 0, 0, 0};
    for (size_t n : {size_t(256), size_t(1000), size_t(100000)}) {
      if (try_decode(wm, 8, n) != 0) {
        fprintf(stderr, "FATAL: wheremeta case accepted at n=%zu\n", n);
        return EXIT_HARNESS_WRONG_RESULT;
      }
    }
    const uint32_t wm512[]    = {512, 0x20000000u, 0, 0, 0, 0, 0, 0};
    const uint32_t bytesize[] = {256, 1, 0xFFFFFFFFu, 0};
    const uint32_t trunc[]    = {256, 1, 0};
    const uint32_t esize[]    = {256, 1, 0, 2, 0xFFFFFFFFu};
    const uint32_t wmzero[]   = {256, 0, 0, 0};
    const uint32_t notblock[] = {1, 0, 0, 0};   // retains coverage of the divisibility throw
    try_decode(wm512, 8, 512);
    try_decode(bytesize, 4, 256);
    try_decode(trunc, 3, 256);
    try_decode(esize, 5, 256);
    try_decode(wmzero, 4, 256);
    try_decode(notblock, 4, 256);
  }

  // ---- 2. valid frames must still round-trip byte-exactly ---------------------------
  {
    // 255 zeroes + one UINT32_MAX: the one-element 32-bit exception stream whose word
    // accounting an over-strict bound would reject.
    std::vector<uint32_t> src(256, 0);
    src[100] = 0xFFFFFFFFu;
    std::vector<uint32_t> blob = encode(src);
    std::vector<uint32_t> out(src.size() + blob.size() * 4);
    size_t got = fastpfor256_decode(blob.data(), blob.size(), out.data(), src.size());
    if (got != src.size() || memcmp(out.data(), src.data(), src.size() * 4) != 0) {
      fprintf(stderr, "FATAL: single-high-exception frame did not round-trip (got %zu)\n", got);
      return EXIT_HARNESS_WRONG_RESULT;
    }
  }
  {
    // multi-page (PageSize is 65536) + a VariableByte tail
    std::mt19937 rng(11);
    std::vector<uint32_t> src(70000);
    for (auto &v : src) v = rng() % 1000000u;
    std::vector<uint32_t> blob = encode(src);
    std::vector<uint32_t> out(src.size() + blob.size() * 4);
    size_t got = fastpfor256_decode(blob.data(), blob.size(), out.data(), src.size());
    if (got != src.size() || memcmp(out.data(), src.data(), src.size() * 4) != 0) {
      fprintf(stderr, "FATAL: multi-page frame did not round-trip (got %zu)\n", got);
      return EXIT_HARNESS_WRONG_RESULT;
    }
  }
  printf("valid round-trips OK\n");

  // ---- 3. exhaustive single-word mutation over a valid frame ------------------------
  // Every word of a well-formed blob replaced by each extreme value, then decoded. This is
  // where an unchecked displacement shows up as a heap-buffer-overflow READ.
  const uint32_t poisons[] = {0u, 1u, 2u, 3u, 0x7Fu, 0xFFu, 0x100u, 0xFFFFu,
                              0x20000000u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu};
  {
    std::mt19937 rng(5);
    std::vector<uint32_t> src(2000);
    for (auto &v : src) v = rng() % 5000u;
    src[7] = 0xFFFFFFFFu;  // force a wide exception stream
    src[900] = 70000u;
    std::vector<uint32_t> blob = encode(src);
    printf("mutation base: %zu values -> %zu words\n", src.size(), blob.size());
    for (size_t i = 0; i < blob.size(); ++i) {
      for (uint32_t p : poisons) {
        std::vector<uint32_t> m = blob;
        m[i] = p;
        try_decode(m.data(), m.size(), src.size());
      }
    }
  }

  // ---- 4. truncation fuzz -- every prefix of a valid frame --------------------------
  {
    std::mt19937 rng(9);
    std::vector<uint32_t> src(1500);
    for (auto &v : src) v = rng() % 100000u;
    std::vector<uint32_t> blob = encode(src);
    for (size_t len = 0; len <= blob.size(); ++len) {
      try_decode(blob.data(), len, src.size());
      try_decode(blob.data(), len, 256);
    }
  }

  // ---- 5. multi-page mutation (sampled) --------------------------------------------
  {
    std::mt19937 rng(13);
    std::vector<uint32_t> src(70000);
    for (auto &v : src) v = rng() % 1000000u;
    std::vector<uint32_t> blob = encode(src);
    for (size_t i = 0; i < blob.size(); i += 37) {
      for (uint32_t p : poisons) {
        std::vector<uint32_t> m = blob;
        m[i] = p;
        try_decode(m.data(), m.size(), src.size());
      }
    }
  }

  // ---- 6. BYTE-granularity mutation -------------------------------------------------
  // The per-block control bytes (b, cexcept, maxbits, exception positions) live packed inside
  // words, so word-level mutation changes four of them at once and can mask a single-field
  // bound. Sweep every byte individually. `cexcept-255` -- the one pre-fix mutation that
  // returned wrong data with no crash and no error -- is a byte-level mutation, so this
  // section is the part of the gate that covers it.
  {
    std::mt19937 rng(21);
    std::vector<uint32_t> src(1024);
    for (auto &v : src) v = rng() % 4000u;
    src[3] = 0xFFFFFFFFu;
    src[500] = 300000u;
    std::vector<uint32_t> blob = encode(src);
    const uint8_t bytepoisons[] = {0u, 1u, 2u, 31u, 32u, 33u, 127u, 255u};
    printf("byte-mutation base: %zu values -> %zu words (%zu bytes)\n", src.size(), blob.size(),
           blob.size() * 4);
    for (size_t bi = 0; bi < blob.size() * 4; ++bi) {
      for (uint8_t p : bytepoisons) {
        std::vector<uint32_t> m = blob;
        reinterpret_cast<uint8_t *>(m.data())[bi] = p;
        try_decode(m.data(), m.size(), src.size());
      }
    }
  }

  printf("decodes=%ld accepted=%ld rejected=%ld\n", g_decodes, g_accepted, g_rejected);
  printf("HARNESS CLEAN\n");
  return 0;
}
