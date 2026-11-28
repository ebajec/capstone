#ifndef ECC_H
#define ECC_H

#include "codec.h"

#include <bit>
#include <numeric>

namespace ecc::detail
{
    constexpr size_t hamming_parity_bits(size_t data_bits)
    {
        size_t p = 0;
        while((1 << p) < data_bits + p + 1)
            ++p;
        return p;
    }

    constexpr size_t hamming_parity_bits_inv(size_t encoded_bits)
    {
        return std::bit_width(encoded_bits);
    }

    template<bool FitsPackedBlock>
    consteval void warn_slow_hamming_code() {}

    template<>
    [[deprecated("Hamming codec uses the slower fallback for this layout")]]
    consteval void warn_slow_hamming_code<false>() {}
} // ecc::detail namespace

// ---------------------------------------------------------------------------------
// ECC interface
struct IErrorCorrectingCode
{
    // Useful function for counting num of encoded/decoded bits given an input size
    virtual constexpr size_t num_encoded_bits(size_t count) const = 0;
    virtual constexpr size_t num_decoded_bits(size_t count) const = 0;
    virtual constexpr const char* name() const = 0;

    constexpr size_t num_encoded_bytes(size_t count) const
    {
        const size_t bits = num_encoded_bits(count);
        return bits / 8 + (bits % 8 != 0);
    }

    constexpr size_t num_decoded_bytes(size_t count) const
    {
        const size_t bits = num_decoded_bits(count);
        return bits / 8 + (bits % 8 != 0);
    }

    virtual void encode(uint8_t* out, size_t count, const uint8_t* in) const = 0;
    virtual void decode(uint8_t* out, size_t count, const uint8_t* in) const = 0;
};
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Repetition codes repeat each input bit three times, then decode each triple by majority.
// Original:  1   0   1
// Encoded:  111 000 111
// Received: 101 001 111
// Decoded:   1   0   1
template<size_t R = 3>
    requires (R > 0 && R <= sizeof(uint64_t))
struct RepetitionCode final : public IErrorCorrectingCode
{
    static constexpr uint8_t rep_mask = (1u << R) - 1u;

    // For repetition codes, the number of encoded and decoded bits its scaled by num_repetitions
    constexpr size_t num_encoded_bits(size_t count) const override { return count * R; }
    constexpr size_t num_decoded_bits(size_t count) const override { return count / R; }
    constexpr const char* name() const override { return "Repetition"; }

    void encode(uint8_t* out, size_t count, const uint8_t* in) const override
    {
        size_t output_byte = 0;
        for (size_t i = 0; i < count; i += 8) // Iterate one byte at a time
        {
            const uint8_t input = in[i / 8];
            PackedWord packed = 0;

            for (uint8_t j = 0; j < 8; ++j)
            {
                const uint8_t bit = (input >> j) & 1;
                const uint8_t repeated_pattern = (0 - bit) & rep_mask;
                packed |= PackedWord(repeated_pattern) << (R * j);
            }

            for (uint8_t j = 0; j < R; ++j) // unpack one byte at a time
                out[output_byte + j] = uint8_t(packed >> (8 * j));
            output_byte += R;
        }
    }

    void decode(uint8_t* out, size_t count, const uint8_t* in) const override
    {
        const size_t out_bytes = num_decoded_bytes(count);

        for (size_t i = 0; i < out_bytes; ++i) // Iterate one byte at a time
        {
            PackedWord packed = 0;
            for (size_t j = 0; j < R; ++j) // Gather all repetitions * bytes worth of encoded bites
                packed |= PackedWord(in[i * R + j]) << (8 * j);

            uint8_t decoded = 0;
            for (uint8_t j = 0; j < 8; ++j) // For each decoded bit, retreieve the repeated pattern
            {
                const uint8_t repeated_pattern = (packed >> (j * R)) & rep_mask;
                const bool bit = std::popcount(repeated_pattern) > R / 2;
                decoded |= uint8_t(bit) << j;
            }

            out[i] = decoded; // write byte
        }
    }

private:
    using PackedWord = std::conditional_t<(R <= sizeof(uint32_t)), uint32_t, uint64_t>;
};
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Hamming code: https://en.wikipedia.org/wiki/Hamming_code
// Needs data and parity bits satisfying: 2^p >= d + p + 1
// Parity bits are placed in positions which are powers of 2 (i.e. 1, 2, 4, 8, etc.)
// Parity bits are used to verify a set of bits within the code and are corrected during decoding
// Example Block Codes:
//          HammingCode<8, 12> (Fastest codec)
//          HammingCode<12, 17>
//          HammingCode<16, 21>
template<size_t D = 8, size_t E = 12>
    requires (D > 0 && D < E && E < 32 && E == D + ecc::detail::hamming_parity_bits(D))
struct HammingCode final : public IErrorCorrectingCode
{
    static constexpr size_t CodewordsPerBlock = std::lcm(8 / std::gcd(8, D), 8 / std::gcd(8, E));
    static constexpr size_t InputBytes = CodewordsPerBlock * D / 8;
    static constexpr size_t OutputBytes = CodewordsPerBlock * E / 8;

    constexpr const char* name() const override { return "Hamming"; }
    constexpr size_t num_encoded_bits(size_t count) const override
    {
        const size_t d = count % D;
        const size_t p = ecc::detail::hamming_parity_bits(d);
        return (count / D) * E + d + p;
    }

    constexpr size_t num_decoded_bits(size_t count) const override
    {
        const size_t n = count % E;
        const size_t p = ecc::detail::hamming_parity_bits_inv(n);
        return (count / E) * D + (n - p);
    }

    void encode(uint8_t* out, size_t count, const uint8_t* in) const override
    {
        size_t input_byte = 0;
        size_t output_byte = 0;
        const size_t block_count = count / (InputBytes * 8);
        constexpr uint64_t data_mask = (uint64_t{1} << D) - 1;

        if constexpr (FitsPackedBlock)
        {
            for(size_t block = 0; block < block_count; block++) // Iterate one block at a time
            {
                PackedWord input = 0;
                for(size_t j = 0; j < InputBytes; j++)
                    input |= PackedWord(in[input_byte + j]) << (8 * j);

                PackedWord encoded = 0;
                for(size_t j = 0; j < CodewordsPerBlock; j++)
                {
                    const PackedWord data = (input >> (D * j)) & data_mask; // Shift by D each time for the data bits
                    encoded |= PackedWord(encodeBlock(data)) << (E * j); // Shift by E each time for the encoded bits
                }

                for(size_t j = 0; j < OutputBytes; j++)
                    out[output_byte + j] = encoded >> (8 * j);

                input_byte += InputBytes;
                output_byte += OutputBytes;
            }
        }

        const size_t remaining = count - input_byte * 8;
        const size_t full_words = remaining / D;
        const size_t tail_bits = remaining % D;

        uint64_t input = 0;
        uint64_t encoded = 0;
        size_t available = 0;
        size_t pending = 0;

        for (size_t word = 0; word < full_words; ++word)
        {
            while (available < D)
            {
                input |= uint64_t(in[input_byte++]) << available;
                available += 8;
            }

            const uint32_t data = uint32_t(input & data_mask);
            input >>= D;
            available -= D;

            encoded |= uint64_t(encodeBlock(data)) << pending;
            pending += E;

            while (pending >= 8)
            {
                out[output_byte++] = uint8_t(encoded);
                encoded >>= 8;
                pending -= 8;
            }
        }

        if (pending != 0)
            out[output_byte] = uint8_t(encoded);

        if (tail_bits != 0)
        {
            uint32_t data;
            read_bits(data, tail_bits, in, count - tail_bits);

            const size_t tail_encoded_bits =
                tail_bits + ecc::detail::hamming_parity_bits(tail_bits);
            write_bits(encodeBlock(data), tail_encoded_bits, out,
                       output_byte * 8 + pending);
        }
    }

    void decode(uint8_t* out, size_t count, const uint8_t* in) const override
    {
        size_t input_byte = 0;
        size_t output_byte = 0;
        constexpr uint64_t codeword_mask = (uint64_t{1} << E) - 1;

        if constexpr (FitsPackedBlock)
        {
            const size_t block_count = count / (OutputBytes * 8);

            for (size_t block = 0; block < block_count; ++block)
            {
                PackedWord encoded = 0;
                for (size_t j = 0; j < OutputBytes; ++j)
                    encoded |= PackedWord(in[input_byte + j]) << (8 * j);

                PackedWord decoded = 0;
                for (size_t j = 0; j < CodewordsPerBlock; ++j)
                {
                    const PackedWord codeword = (encoded >> (E * j)) & codeword_mask;
                    decoded |= PackedWord(decodeBlock(codeword)) << (D * j);
                }

                for (size_t j = 0; j < InputBytes; ++j)
                    out[output_byte + j] = decoded >> (8 * j);

                input_byte += OutputBytes;
                output_byte += InputBytes;
            }
        }

        const size_t remaining = count - input_byte * 8;
        const size_t full_words = remaining / E;
        const size_t tail_bits = remaining % E;

        uint64_t encoded = 0;
        uint64_t decoded = 0;
        size_t available = 0;
        size_t pending = 0;

        for (size_t word = 0; word < full_words; ++word)
        {
            while (available < E)
            {
                encoded |= uint64_t(in[input_byte++]) << available;
                available += 8;
            }

            const uint32_t codeword = uint32_t(encoded & codeword_mask);
            encoded >>= E;
            available -= E;

            decoded |= uint64_t(decodeBlock(codeword)) << pending;
            pending += D;

            while (pending >= 8)
            {
                out[output_byte++] = uint8_t(decoded);
                decoded >>= 8;
                pending -= 8;
            }
        }

        if (pending != 0)
            out[output_byte] = uint8_t(decoded);

        const size_t tail_data_bits =
            tail_bits - ecc::detail::hamming_parity_bits_inv(tail_bits);

        if (tail_data_bits != 0)
        {
            uint32_t codeword;
            read_bits(codeword, tail_bits, in, count - tail_bits);
            write_bits(decodeBlock(codeword), tail_data_bits,
                       out, output_byte * 8 + pending);
        }
    }
private:
    using PackedWord = std::conditional_t<(OutputBytes <= sizeof(uint32_t)), uint32_t, uint64_t>;

    static constexpr bool FitsPackedBlock = [] {
        constexpr bool fits_packed_block = OutputBytes <= sizeof(PackedWord);
        ecc::detail::warn_slow_hamming_code<fits_packed_block>(); // Warn against slow hamming codes
        return fits_packed_block;
    }();

    static constexpr uint32_t encodeBlock(uint32_t data)
    {
        uint32_t out = 0;

        size_t d_idx = 0;
        for(size_t output_offset = 0; output_offset < E; output_offset++)
        {
            const size_t output_location = output_offset + 1;
            if((output_location & (output_location - 1)) != 0) // Is power of two; skip parity bit
            {
                const uint32_t bit = (data >> d_idx) & 1;
                out |= bit << output_offset;
                d_idx++;
            }
        }

        for(size_t parity_pos = 1; parity_pos <= E; parity_pos <<= 1)
        {
            uint32_t parity_bit = 0;

            for(size_t data_pos = parity_pos; data_pos <= E; ++data_pos)
            {
                if(parity_pos & data_pos) // Find all bits this parity bit accounts for
                {
                    const uint32_t bit = (out >> (data_pos - 1)) & 1;
                    parity_bit ^= bit;
                }
            }

            // An odd number of ones needs the parity bit set.
            if(parity_bit)
            {
                out |= 1 << (parity_pos - 1);
            }
        }

        return out;
    }

    static constexpr uint32_t decodeBlock(uint32_t encoded)
    {
        // Check parity groups
        size_t error_position = 0;
        for(size_t parity_pos = 1; parity_pos <= E; parity_pos <<= 1)
        {
            uint32_t parity_bit = 0;

            for(size_t data_pos = parity_pos; data_pos <= E; ++data_pos)
            {
                if(data_pos & parity_pos)
                {
                    const uint32_t bit = (encoded >> (data_pos - 1)) & 1;
                    parity_bit ^= bit;
                }
            }

            if(parity_bit)
            {
                error_position |= parity_pos;
            }
        }

        uint32_t out = 0;
        size_t output_offset = 0;
        for(size_t i = 0; i < E; ++i )
        {
            const size_t pos = i + 1;
            uint32_t bit = (encoded >> i) & 1;

            if(pos == error_position)
            {
                bit ^= 1;
            }

            if((pos & (pos - 1)) != 0) // Prevent copying parity bits
            {
                out |= bit << output_offset;
                output_offset++;
            }
        }

        return out;
    }
};
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// BCH code:
// struct BCHCode final : public IErrorCorrectingCode
// {
//     static constexpr size_t q = 2;
//     static constexpr size_t
//
//     constexpr size_t num_encoded_bits(size_t count) const override;
//     constexpr size_t num_decoded_bits(size_t count) const override;
//     constexpr const char* name() const override { return "BCH"; }
//
//     void encode(uint8_t* out, size_t count, const uint8_t* in) const override;
//     void decode(uint8_t* out, size_t count, const uint8_t* in) const override;
// };
// ---------------------------------------------------------------------------------

#endif
