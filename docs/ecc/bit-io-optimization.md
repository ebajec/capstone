# Packed bit I/O optimization

The repetition codec showed that assembling a block in a register and reading or writing whole bytes can be much faster than visiting the same bytes one bit at a time. Apply that approach to the shared [bit helpers](../../src/shared/codec.h) while keeping their LSB-first bit order and support for fields that cross byte boundaries. This is follow-up work. The byte-at-a-time `read_bits` prototype was benchmarked and reverted.

## What the repetition codec showed

One input byte expands to `8 * repetitions` encoded bits, or exactly `repetitions` bytes. The [encoder](../../src/error/main.cpp) creates each repeated field with `(0u - bit) & rep_mask`, places it at `j * repetitions` in a register, then stores the output bytes. The decoder gathers those bytes into a register, extracts eight fields, and applies a majority vote with `std::popcount(group) > repetitions / 2`. A 32-bit accumulator holds up to four repetitions per input bit. A 64-bit accumulator is required for factors five through eight, and the field must be widened before shifting. The byte-at-a-time codec was benchmarked on whole input bytes. Partial blocks need separate handling if that codec is used for arbitrary bit counts.

Apple Clang 21 on ARM64 at `-O3` unrolled the fixed-size packing loop and the 24-bit writer loop. The generic writer still emitted dependent load–OR–store sequences because adjacent iterations update the same output byte through addresses of the form `bytes[(offset + j) >> 3]`. Passing an advanced pointer and a literal offset of zero made the byte indices constant in the unrolled body. Clang then merged the updates into three stores. Direct stores express that operation immediately. Packing still has a dependency through the register accumulator.

| 1 MiB Release benchmark | Median time |
| --- | ---: |
| Repetition encode with packed 24-bit words and a running bit offset passed to `write_bits` | 5.33 ms |
| Repetition encode with packed words and `write_bits` called with an advanced pointer and offset zero | 1.19 ms |
| Repetition encode with packed words and three direct byte stores | 0.60 ms |
| Hamming decode with the original `read_bits` | 12.43 ms |
| Hamming decode with the byte-at-a-time `read_bits` prototype | 8.96 ms |

The prototypes passed the codec round trips. Release is the useful comparison for arithmetic types here. Narrow locals added byte spills and reloads in Debug, while both tested integer widths produced the same input and output memory operation counts at `-O3`.

## Read bits one byte at a time

For a positive width of up to 32 bits, compute `first_byte = offset >> 3`, `first_bit = offset & 7`, and `byte_count = (first_bit + nbits + 7) >> 3`. Load exactly those bytes into a `uint64_t` at shifts `0, 8, 16, ...`. Shift right by `first_bit`, keep the low `nbits` bits, and return `offset + nbits`. A 32-bit field can span five bytes. Handle width zero before loading and use a mask expression that works for width 32. The prototype passed ASan and UBSan checks for widths 0–32 with exact-sized buffers, including unaligned fields.

## Write bits one byte at a time

Mask the source word to its field width before shifting it left by `first_bit` in a 64-bit temporary. Preserve the bits below `first_bit` in the first output byte with one masked read when the offset is unaligned. Assign each touched output byte once from the temporary. Masking the source before shifting preserves the existing zero padding above the field in the final touched byte. Leave bytes outside the touched range alone. Characterize the zero-width case separately because the current writer modifies the starting byte even for a zero-bit write.

Expose a literal zero bit offset or a byte-aligned overload for fixed-width callers. Keep the generic path for fields at arbitrary offsets. Check exact byte output against the current writer using prefilled destinations, all eight starting alignments, and widths through 32. Benchmark byte-aligned 8- and 24-bit fields, 12-bit Hamming fields at alternating offsets, and variable-width fields. Use Release assembly to check that the aligned path emits direct stores and that the unaligned path only reads the first destination byte when it needs to preserve its low bits.
