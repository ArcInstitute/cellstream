/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire, http://lemire.me/en/
 */
#ifndef SIMDFASTPFOR_H_
#define SIMDFASTPFOR_H_

#include "codecs.h"
#include "common.h"
#include "memutil.h"
#include "simdbitpacking.h"
#include "simple8b.h"
#include "usimdbitpacking.h"
#include "util.h"
#include <stdexcept>   // project issue #269: bounds-check throws

namespace FastPForLib {

/**
 * SIMDFastPFor
 *
 * In a multithreaded context, you may need one SIMDFastPFor per thread.
 *
 *
 * Reference and documentation:
 *
 * Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second
 * through std::vectorization
 * Software: Practice & Experience
 * http://arxiv.org/abs/1209.2137
 * http://onlinelibrary.wiley.com/doi/10.1002/spe.2203/abstract
 *
 * Note: the algorithms were slightly revised in Sept. 2013 to improve the
 * compression
 * ratios. You can expect the same compression ratios as the scalar FastPFOR (up
 * to a difference of 1%).
 *
 * Designed by D. Lemire with ideas from Leonid Boytsov. This scheme is NOT
 * patented.
 *
 */
template <uint32_t BlockSizeInUnitsOfPackSize = 8> // BlockSizeInUnitsOfPackSize can have value 4 or 8
class SIMDFastPFor : public IntegerCODEC {
public:
  using IntegerCODEC::decodeArray;
  using IntegerCODEC::encodeArray;

  /**
   * ps (page size) should be a multiple of BlockSize, any "large"
   * value should do.
   */
  SIMDFastPFor(uint32_t ps = 65536)
      : PageSize(ps), bitsPageSize(gccbits(PageSize)), datatobepacked(33),
        bytescontainer(PageSize + 3 * PageSize / BlockSize) {
    assert(ps / BlockSize * BlockSize == ps);
    assert(gccbits(static_cast<uint32_t>(BlockSizeInUnitsOfPackSize * PACKSIZE - 1)) <= 8);
  }
  enum {
    PACKSIZE = 32,
    overheadofeachexcept = 8,
    overheadduetobits = 8,
    overheadduetonmbrexcept = 8,
    BlockSize = BlockSizeInUnitsOfPackSize * PACKSIZE
  };

  static uint32_t *packblockupsimd(const uint32_t *source, uint32_t *out, const uint32_t bit) {
    for (int k = 0; k < BlockSize; k += 128) {
      SIMD_fastpack_32(source, reinterpret_cast<__m128i *>(out), bit);
      out += 4 * bit;
      source += 128;
    }
    return out;
  }

  static const uint32_t *unpackblocksimd(const uint32_t *source, uint32_t *out, const uint32_t bit) {
    for (int k = 0; k < BlockSize; k += 128) {
      SIMD_fastunpack_32(reinterpret_cast<const __m128i *>(source), out, bit);
      source += 4 * bit;
      out += 128;
    }
    return source;
  }

  template <class STLContainer>
  static const uint32_t *unpackmesimd(const uint32_t *in, STLContainer &out, const uint32_t bit) {
    const uint32_t size = *in;
    ++in;
    out.resize((size + 32 - 1) / 32 * 32);
    uint32_t j = 0;
    for (; j + 128 <= size; j += 128) {
      usimdunpack(reinterpret_cast<const __m128i *>(in), &out[j], bit);
      in += 4 * bit;
    }
    for (; j + 31 < size; j += 32) {
      fastunpack(in, &out[j], bit);
      in += bit;
    }
    uint32_t buffer[PACKSIZE] = {};   // project issue #269
    const uint32_t remaining = size - j;
    memcpy(buffer, in, (remaining * bit + 31) / 32 * sizeof(uint32_t));
    uint32_t *bpointer = buffer;
    // project issue #269: advance by what was consumed; the old form advanced by k and backed up,
    // transiently forming an out-of-object pointer.
    in += (remaining * bit + 31) / 32;
    for (; j < size; j += 32) {
      fastunpack(bpointer, &out[j], bit);
      bpointer += bit;
    }
    out.resize(size);
    return in;
  }

  template <class STLContainer>
  static uint32_t *packmeupwithoutmasksimd(STLContainer &source, uint32_t *out, const uint32_t bit) {
    const uint32_t size = static_cast<uint32_t>(source.size());
    *out = size;
    out++;
    if (source.size() == 0)
      return out;
    source.resize((source.size() + 32 - 1) / 32 * 32);
    uint32_t j = 0;
    for (; j + 128 <= size; j += 128) {
      usimdpackwithoutmask(&source[j], reinterpret_cast<__m128i *>(out), bit);
      out += 4 * bit;
    }
    for (; j < size; j += 32) {
      fastpackwithoutmask(&source[j], out, bit);
      out += bit;
    }
    out -= (j - size) * bit / 32;
    source.resize(size);
    return out;
  }

  // sometimes, mem. usage can grow too much, this clears it up
  void resetBuffer() {
    for (size_t i = 0; i < datatobepacked.size(); ++i) {
      std::vector<uint32_t, cacheallocator>().swap(datatobepacked[i]);
    }
  }

  const uint32_t PageSize;
  const uint32_t bitsPageSize;

  std::vector<std::vector<uint32_t, cacheallocator>> datatobepacked;
  std::vector<uint8_t> bytescontainer;

  // project issue #269: `length` is now ALWAYS named -- it is the only bound available to
  // __decodeArray, whose displacements all come from the (untrusted) blob. Deleting the
  // NDEBUG split is deliberate: upstream named `length` only under `#ifndef NDEBUG` and used
  // it solely for a trailing assert, so -DNDEBUG silently dropped the bound. It is now
  // load-bearing in ordinary code (`inend` below), which NDEBUG cannot remove. Verified by
  // `rust/asan/run.sh`, which gates this header both with assertions live and with -DNDEBUG.
  const uint32_t *decodeArray(const uint32_t *in, const size_t length,
                              uint32_t *out, size_t &nvalue) override {
    const uint32_t *const initin(in);
    const uint32_t *const inend(initin + length);
    if (length == 0)
      throw std::runtime_error("FastPFor: empty input");
    const size_t mynvalue = *in;
    ++in;
    if (mynvalue > nvalue)
      throw NotEnoughStorage(mynvalue);
    // project issue #269: CompositeCodec gives codec1 only the BlockSize-aligned prefix, so a header
    // claiming otherwise would make the page loop advance `out` past what __decodeArray wrote,
    // exposing uninitialised memory as decoded values.
    if (mynvalue % BlockSize != 0)
      throw std::runtime_error("FastPFor: decoded count not a multiple of BlockSize");
    nvalue = mynvalue;
    const uint32_t *const finalout(out + nvalue);
    while (out != finalout) {
      size_t thisnvalue(0);
      // project issue #269: `PageSize + out` formed an out-of-object pointer (UB) on small outputs.
      size_t thissize =
          static_cast<size_t>(finalout - out > static_cast<ptrdiff_t>(PageSize)
                                  ? PageSize
                                  : (finalout - out));
      __decodeArray(in, thisnvalue, out, thissize, inend);
      in += thisnvalue;
      out += thissize;
    }
    assert(initin + length >= in);
    resetBuffer(); // if you don't do this, the codec has a "memory".
    return in;
  }

  /**
   * If you save the output and recover it in memory, you are
   * responsible to ensure that the alignment is preserved.
   *
   * The input size (length) should be a multiple of
   * BlockSizeInUnitsOfPackSize * PACKSIZE. (This was done
   * to simplify slightly the implementation.)
   */
  void encodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) override {
    checkifdivisibleby(length, BlockSize);
#ifndef NDEBUG
    const uint32_t *const initout(out);
#endif
    const uint32_t *const finalin(in + length);

    *out++ = static_cast<uint32_t>(length);
    const size_t oldnvalue = nvalue;
    nvalue = 1;
    while (in != finalin) {
      size_t thissize = static_cast<size_t>(finalin > PageSize + in ? PageSize : (finalin - in));
      size_t thisnvalue(0);
      __encodeArray(in, thissize, out, thisnvalue);
      nvalue += thisnvalue;
      out += thisnvalue;
      in += thissize;
    }
    assert(out == nvalue + initout);
    if (oldnvalue < nvalue)
      fprintf(stderr,
              "It is possible we have a buffer overrun. You reported having allocated "
              "%zu bytes for the compressed data but we needed "
              "%zu bytes. Please increase the available memory "
              "for compressed data or check the value of the last parameter provided "
              "to the encodeArray method.\n",
              oldnvalue * sizeof(uint32_t), nvalue * sizeof(uint32_t));
    resetBuffer(); // if you don't do this, the buffer has a memory
  }

  void getBestBFromData(const uint32_t *in, uint8_t &bestb, uint8_t &bestcexcept, uint8_t &maxb) {
    uint32_t freqs[33];
    for (uint32_t k = 0; k <= 32; ++k)
      freqs[k] = 0;
    for (uint32_t k = 0; k < BlockSize; ++k) {
      freqs[asmbits(in[k])]++;
    }
    bestb = 32;
    while (freqs[bestb] == 0)
      bestb--;
    maxb = bestb;
    uint32_t bestcost = bestb * BlockSize;
    uint32_t cexcept = 0;
    bestcexcept = static_cast<uint8_t>(cexcept);
    for (uint32_t b = bestb - 1; b < 32; --b) {
      cexcept += freqs[b + 1];
      uint32_t thiscost = cexcept * overheadofeachexcept + cexcept * (maxb - b) + b * BlockSize +
                          8; // the  extra 8 is the cost of storing maxbits
      if (thiscost < bestcost) {
        bestcost = thiscost;
        bestb = static_cast<uint8_t>(b);
        bestcexcept = static_cast<uint8_t>(cexcept);
      }
    }
  }

  void __encodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
    uint32_t *const initout = out; // keep track of this
    checkifdivisibleby(length, BlockSize);
    uint32_t *const headerout = out++; // keep track of this
    for (uint32_t k = 0; k < 32 + 1; ++k)
      datatobepacked[k].clear();
    uint8_t *bc = &bytescontainer[0];
    for (const uint32_t *const final = in + length; (in + BlockSize <= final); in += BlockSize) {
      uint8_t bestb, bestcexcept, maxb;
      getBestBFromData(in, bestb, bestcexcept, maxb);
      *bc++ = bestb;
      *bc++ = bestcexcept;
      if (bestcexcept > 0) {
        *bc++ = maxb;
        std::vector<uint32_t, cacheallocator> &thisexceptioncontainer = datatobepacked[maxb - bestb];
        const uint32_t maxval = 1U << bestb;
        for (uint32_t k = 0; k < BlockSize; ++k) {
          if (in[k] >= maxval) {
            // we have an exception
            thisexceptioncontainer.push_back(in[k] >> bestb);
            *bc++ = static_cast<uint8_t>(k);
          }
        }
      }
      out = packblockupsimd(in, out, bestb);
    }
    headerout[0] = static_cast<uint32_t>(out - headerout);
    const uint32_t bytescontainersize = static_cast<uint32_t>(bc - &bytescontainer[0]);
    *(out++) = bytescontainersize;
    memcpy(out, &bytescontainer[0], bytescontainersize);
    uint8_t *pad8 = (uint8_t *)out + bytescontainersize;
    out += (bytescontainersize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    while (pad8 < (uint8_t *)out)
      *pad8++ = 0; // clear padding bytes

    uint32_t bitmap = 0;
    for (uint32_t k = 2; k <= 32; ++k) {
      if (datatobepacked[k].size() != 0)
        bitmap |= (1U << (k - 1));
    }
    *(out++) = bitmap;
    for (uint32_t k = 2; k <= 32; ++k) {
      if (datatobepacked[k].size() > 0)
        out = packmeupwithoutmasksimd(datatobepacked[k], out, k);
    }
    nvalue = out - initout;
  }

  void __decodeArray(const uint32_t *in, size_t &length, uint32_t *out, const size_t nvalue,
                     const uint32_t *const inend) {
    // project issue #269: EVERY displacement below is taken from the blob, which is untrusted.
    // Bound each and throw; rust/csrc/pfor_shim.cpp's catch(...) turns the throw into a 0
    // return, which ffi_decode reports as a clean corrupt-frame ValueError.
    const uint32_t *const initin = in;
    if (in >= inend)
      throw std::runtime_error("FastPFor: truncated block header");
    const uint32_t *const headerin = in++;
    const uint32_t wheremeta = headerin[0];
    // wheremeta is the END of the packed blocks and the START of the metadata.
    if (wheremeta < 1 || static_cast<size_t>(wheremeta) >= static_cast<size_t>(inend - headerin))
      throw std::runtime_error("FastPFor: wheremeta out of range");
    const uint32_t *const packedend = headerin + wheremeta;
    const uint32_t *inexcept = packedend;
    const uint32_t bytesize = *inexcept++;
    const uint8_t *bytep = reinterpret_cast<const uint8_t *>(inexcept);
    const size_t byteadvance =
        (static_cast<size_t>(bytesize) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    if (byteadvance > static_cast<size_t>(inend - inexcept))
      throw std::runtime_error("FastPFor: exception byte block out of range");
    const uint8_t *const bytepend = bytep + bytesize;
    inexcept += byteadvance;
    if (inexcept >= inend)
      throw std::runtime_error("FastPFor: bitmap out of range");
    const uint32_t bitmap = *(inexcept++);
    bool present[32 + 1] = {};
    size_t used[32 + 1] = {};
    size_t total_exceptions = 0;
    for (uint32_t k = 2; k <= 32; ++k) {
      if ((bitmap & (1U << (k - 1))) != 0) {
        if (inexcept >= inend)
          throw std::runtime_error("FastPFor: exception block size out of range");
        const uint32_t esize = *inexcept;
        // CORRECTED accounting. unpackmesimd consumes the size word plus ceil(esize*k/32)
        // words: bit words per 32 values across both loops, then ceil(remaining*k/32) for the
        // memcpy tail. The earlier `1 + ceil(esize/32)*k` form OVER-counts (esize=1,k=32 -> 33
        // words demanded vs 2 actually consumed) and would reject valid archives.
        // size_t before the +31 -- `(esize + 31)` wraps in uint32_t at esize=0xffffffff.
        // validate BEFORE the arithmetic: on a 64-bit size_t esize*k peaks near 2^37 and cannot
        // overflow, but ordering the check first holds on a 32-bit size_t too and is free.
        if (esize == 0 || esize > nvalue)
          throw std::runtime_error("FastPFor: exception count out of range");
        const size_t data_words = (static_cast<size_t>(esize) * k + 31) / 32;
        total_exceptions += esize;
        if (total_exceptions > nvalue)
          throw std::runtime_error("FastPFor: exception counts exceed the page");
        if (1 + data_words > static_cast<size_t>(inend - inexcept))
          throw std::runtime_error("FastPFor: exception block out of range");
        inexcept = unpackmesimd(inexcept, datatobepacked[k], k);
        present[k] = true;
      }
    }
    length = inexcept - initin;
    std::vector<uint32_t, cacheallocator>::const_iterator unpackpointers[32 + 1];
    for (uint32_t k = 1; k <= 32; ++k) {
      unpackpointers[k] = datatobepacked[k].begin();
    }
    for (uint32_t run = 0; run < nvalue / BlockSize; ++run, out += BlockSize) {
      // pointer-difference form: `bytep + 2` may itself be an out-of-object pointer (UB).
      if (static_cast<size_t>(bytepend - bytep) < 2)
        throw std::runtime_error("FastPFor: exception bytes exhausted");
      const uint8_t b = *bytep++;
      const uint8_t cexcept = *bytep++;
      if (b > 32)
        throw std::runtime_error("FastPFor: bad bit width");
      if (static_cast<size_t>(BlockSize) * b / 32 > static_cast<size_t>(packedend - in))
        throw std::runtime_error("FastPFor: packed block out of range");
      in = unpackblocksimd(in, out, b);
      if (cexcept > 0) {
        if (static_cast<size_t>(bytepend - bytep) < 1 + static_cast<size_t>(cexcept))
          throw std::runtime_error("FastPFor: exception bytes exhausted");
        const uint8_t maxbits = *bytep++;
        // `maxbits - b` indexes unpackpointers[]; both bytes are untrusted, and maxbits < b
        // promotes to a huge index.
        if (maxbits <= b || maxbits > 32)
          throw std::runtime_error("FastPFor: bad exception bit width");
        if (maxbits - b == 1) {
          for (uint32_t k = 0; k < cexcept; ++k) {
            const uint8_t pos = *(bytep++);
            if (static_cast<size_t>(pos) >= static_cast<size_t>(BlockSize))
              throw std::runtime_error("FastPFor: exception position out of range");
            out[pos] |= static_cast<uint32_t>(1) << b;
          }
        } else {
          const uint32_t diff = static_cast<uint32_t>(maxbits) - static_cast<uint32_t>(b);
          if (!present[diff])
            throw std::runtime_error("FastPFor: exception stream absent for this page");
          if (used[diff] + cexcept > datatobepacked[diff].size())
            throw std::runtime_error("FastPFor: exception stream exhausted");
          used[diff] += cexcept;
          std::vector<uint32_t, cacheallocator>::const_iterator &exceptionsptr =
              unpackpointers[diff];
          for (uint32_t k = 0; k < cexcept; ++k) {
            const uint8_t pos = *(bytep++);
            if (static_cast<size_t>(pos) >= static_cast<size_t>(BlockSize))
              throw std::runtime_error("FastPFor: exception position out of range");
            out[pos] |= (*(exceptionsptr++)) << b;
          }
        }
      }
    }
    // project issue #269: was `assert(...)`, which ABORTS the process on a corrupt frame instead of
    // producing a catchable error -- and vanishes entirely under -DNDEBUG.
    if (in != packedend)
      throw std::runtime_error("FastPFor: packed block length mismatch");
  }

  std::string name() const override { return std::string("SIMDFastPFor") + std::to_string(BlockSize); }
};

/**
 * SIMDSimplePFor: similar to SimplePFor but with SIMD acceleration.
 *
 *
 * Designed by D. Lemire. This scheme is NOT patented.
 *
 * Reference and documentation:
 *
 * Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second
 * through std::vectorization
 * http://arxiv.org/abs/1209.2137
 *
 */
template <class EXCEPTIONCODER = Simple8b<true>> class SIMDSimplePFor : public IntegerCODEC {
public:
  using IntegerCODEC::decodeArray;
  using IntegerCODEC::encodeArray;

  EXCEPTIONCODER ecoder;
  /**
   * ps (page size) should be a multiple of BlockSize, any "large"
   * value should do.
   */
  SIMDSimplePFor(uint32_t ps = 65536)
      : ecoder(), PageSize(ps), bitsPageSize(gccbits(PageSize)), datatobepacked(PageSize),
        bytescontainer(PageSize + 3 * PageSize / BlockSize) {
    assert(ps / BlockSize * BlockSize == ps);
    assert(gccbits(static_cast<uint32_t>(BlockSizeInUnitsOfPackSize * PACKSIZE - 1)) <= 8);
  }
  enum {
    BlockSizeInUnitsOfPackSize = 4,
    PACKSIZE = 32,
    overheadofeachexcept = 8,
    overheadduetobits = 8,
    overheadduetonmbrexcept = 8,
    BlockSize = BlockSizeInUnitsOfPackSize * PACKSIZE
  };

  const uint32_t PageSize;
  const uint32_t bitsPageSize;

  std::vector<uint32_t> datatobepacked;
  std::vector<uint8_t> bytescontainer;

  const uint32_t *decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) override {
    const uint32_t *const initin(in);
    const size_t mynvalue = *in;
    ++in;
    if (mynvalue > nvalue)
      throw NotEnoughStorage(mynvalue);
    nvalue = mynvalue;
    const uint32_t *const finalout(out + nvalue);
    while (out != finalout) {
      size_t thisnvalue = length - (in - initin);
      size_t thissize = static_cast<size_t>(finalout > PageSize + out ? PageSize : (finalout - out));

      __decodeArray(in, thisnvalue, out, thissize);
      in += thisnvalue;
      out += thissize;
    }
    assert(initin + length >= in);
    return in;
  }

  /*
   * The input size (length) should be a multiple of
   * BlockSizeInUnitsOfPackSize * PACKSIZE. (This was done
   * to simplify slightly the implementation.)
   */
  void encodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) override {
    checkifdivisibleby(length, BlockSize);
    const uint32_t *const initout(out);
    const uint32_t *const finalin(in + length);

    *out++ = static_cast<uint32_t>(length);
    const size_t oldnvalue = nvalue;
    nvalue = 1;
    while (in != finalin) {
      size_t thissize = static_cast<size_t>(finalin > PageSize + in ? PageSize : (finalin - in));
      size_t thisnvalue = oldnvalue - (out - initout);
      __encodeArray(in, thissize, out, thisnvalue);
      nvalue += thisnvalue;
      out += thisnvalue;
      in += thissize;
    }
    assert(out == nvalue + initout);
    if (oldnvalue < nvalue)
      fprintf(stderr,
              "It is possible we have a buffer overrun. You reported having allocated "
              "%zu bytes for the compressed data but we needed "
              "%zu bytes. Please increase the available memory "
              "for compressed data or check the value of the last parameter provided "
              "to the encodeArray method.\n",
              oldnvalue * sizeof(uint32_t), nvalue * sizeof(uint32_t));
  }

  void getBestBFromData(const uint32_t *in, uint8_t &bestb, uint8_t &bestcexcept, uint8_t &maxb) {
    uint32_t freqs[33];
    for (uint32_t k = 0; k <= 32; ++k)
      freqs[k] = 0;
    for (uint32_t k = 0; k < BlockSize; ++k) {
      freqs[asmbits(in[k])]++;
    }
    bestb = 32;
    while (freqs[bestb] == 0)
      bestb--;
    maxb = bestb;
    uint32_t bestcost = bestb * BlockSize;
    uint32_t cexcept = 0;
    bestcexcept = static_cast<uint8_t>(cexcept);
    for (uint32_t b = bestb - 1; b < 32; --b) {
      cexcept += freqs[b + 1];
      uint32_t thiscost = cexcept * overheadofeachexcept + cexcept * (maxb - b) + b * BlockSize;
      if (thiscost < bestcost) {
        bestcost = thiscost;
        bestb = static_cast<uint8_t>(b);
        bestcexcept = static_cast<uint8_t>(cexcept);
      }
    }
  }

  void __encodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
    uint32_t *const initout = out; // keep track of this
    checkifdivisibleby(length, BlockSize);
    uint32_t *const headerout = out++; // keep track of this
    datatobepacked.clear();
    uint8_t *bc = &bytescontainer[0];
    for (const uint32_t *const final = in + length; (in + BlockSize <= final); in += BlockSize) {
      uint8_t bestb, bestcexcept, maxb;
      getBestBFromData(in, bestb, bestcexcept, maxb);
      *bc++ = bestb;
      *bc++ = bestcexcept;
      if (bestcexcept > 0) {
        assert(bestb < 32);
        const uint32_t maxval = 1U << bestb;
        for (uint32_t k = 0; k < BlockSize; ++k) {
          if (in[k] >= maxval) {
            datatobepacked.push_back(in[k] >> bestb);
            *bc++ = static_cast<uint8_t>(k);
          }
        }
      }
      assert(BlockSize == 128);
      usimdpack(in, reinterpret_cast<__m128i *>(out), bestb);
      out += BlockSizeInUnitsOfPackSize * bestb;
      // out = packblockup<BlockSize>(in, out, bestb);
    }
    headerout[0] = static_cast<uint32_t>(out - headerout);
    const uint32_t bytescontainersize = static_cast<uint32_t>(bc - &bytescontainer[0]);
    *(out++) = bytescontainersize;
    memcpy(out, &bytescontainer[0], bytescontainersize);
    uint8_t *pad8 = (uint8_t *)out + bytescontainersize;
    out += (bytescontainersize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    while (pad8 < (uint8_t *)out)
      *pad8++ = 0; // clear padding bytes

    size_t outcap = 0;
    ecoder.encodeArray(datatobepacked.data(), datatobepacked.size(), out, outcap);
    out += outcap;
    nvalue = out - initout;
  }

  void __decodeArray(const uint32_t *in, size_t &length, uint32_t *out, const size_t nvalue) {
    const uint32_t *const initin = in;
    const uint32_t *const headerin = in++;
    const uint32_t wheremeta = headerin[0];
    const uint32_t *inexcept = headerin + wheremeta;
    const uint32_t bytesize = *inexcept++;
    const uint8_t *bytep = reinterpret_cast<const uint8_t *>(inexcept);
    inexcept += (bytesize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    datatobepacked.resize(datatobepacked.capacity());
    size_t cap = datatobepacked.size();
    size_t le = initin + length - inexcept;
    inexcept = ecoder.decodeArray(inexcept, le, &datatobepacked[0], cap);

    length = inexcept - initin;

    auto exceptionsptr = datatobepacked.begin();
    for (uint32_t run = 0; run < nvalue / BlockSize; ++run, out += BlockSize) {
      const uint8_t b = *bytep++;
      const uint8_t cexcept = *bytep++;
      // in = unpackblock<BlockSize>(in, out, b);
      assert(BlockSize == 128);
      usimdunpack(reinterpret_cast<const __m128i *>(in), out, b);
      in += BlockSizeInUnitsOfPackSize * b;
      for (uint32_t k = 0; k < cexcept; ++k) {
        const uint8_t pos = *(bytep++);
        out[pos] |= (*(exceptionsptr++)) << b;
      }
    }
    assert(in == headerin + wheremeta);
  }

  std::string name() const override { return "SIMDSimplePFor"; }
};

} // namespace FastPForLib

#endif /* SIMDFASTPFOR_H_ */
