#include "codec.h"

#include <cstdio>
#include <filesystem>
#include <vector>

using Bytes = std::vector<std::uint8_t>;

struct BitBuffer
{
    Bytes bytes;
    size_t bit_count = 0;

    std::string toString() const;
};

std::string BitBuffer::toString() const
{
    std::string result;
    
    for (size_t i = 0; i < bit_count; ++i)
    {
        uint32_t bit = 0;
        read_bits(bit, 1, bytes.data(), i);
        result += std::to_string(bit);
    }

    return result;
}

struct IErrorCorrectingCode
{
    virtual BitBuffer encode(const BitBuffer& in) const = 0;
    virtual BitBuffer decode(const BitBuffer& in) const = 0;
};

// Repeat each input bit three times, then decode each triple by majority.
// Original:  1   0   1
// Encoded:  111 000 111
// Received: 101 001 111
// Decoded:   1   0   1
struct RepetitionCode final : public IErrorCorrectingCode
{
    static constexpr uint32_t repetitions = 3;

    BitBuffer encode(const BitBuffer& in) const override;
    BitBuffer decode(const BitBuffer& in) const override;
};

// Hamming code: https://en.wikipedia.org/wiki/Hamming_code
// Needs data and parity bits satisfying: 2^p >= d + p + 1
// Parity bits are placed in positions which are powers of 2 (i.e. 1, 2, 4, 8, etc.)
//
struct HammingCode final : public IErrorCorrectingCode
{
    BitBuffer encode(const BitBuffer& in) const override;
    BitBuffer decode(const BitBuffer& in) const override;
};

BitBuffer RepetitionCode::encode(const BitBuffer& in) const
{
    BitBuffer out;
    out.bit_count = in.bit_count * repetitions;
    out.bytes.resize(out.bit_count, 0);

    size_t output_offset = 0;
    for (size_t i = 0; i < in.bit_count; ++i)
    {
        uint32_t bit = 0;
        read_bits(bit, 1, in.bytes.data(), i); // Read one bit at a time

        for (uint32_t j = 0; j < repetitions; ++j)
            output_offset = write_bits(bit, 1, out.bytes.data(), output_offset);
    }

    return out;
}

BitBuffer RepetitionCode::decode(const BitBuffer& in) const
{
    BitBuffer out;
    out.bit_count = in.bit_count / repetitions;
    out.bytes.resize(out.bit_count, 0);

    size_t input_offset = 0;
    for (size_t i = 0; i < out.bit_count; ++i)
    {
        uint32_t ones = 0;
        for (uint32_t j = 0; j < repetitions; ++j)
        {
            uint32_t bit = 0;
            input_offset = read_bits(bit, 1, in.bytes.data(), input_offset);
            ones += bit; // count number of ones
        }

        // If we have more ones than the number of repetitions over 2, then we decode a 1, otherwise 0
        const uint32_t bit = ones > repetitions / 2 ? 1 : 0;
        write_bits(bit, 1, out.bytes.data(), i);
    }

    return out;
}


BitBuffer HammingCode::encode(const BitBuffer& in) const
{
    BitBuffer out;
    out.bytes.resize(in.bit_count, 0);
    out.bit_count = in.bit_count;

    // Satify the constraint: 2^p >= d + p + 1
    Bytes p; // Determine parity bits

    size_t input_offset = 0;
    for(size_t i = 0; i < in.bit_count; ++i)
    {
        if(auto bit_location = i + 1; (bit_location & (bit_location - 1)) == 0) // Is power of two
        {
            out.bytes.push_back(0);
            out.bit_count++;

            uint32_t bit = 0;
            write_bits(bit, 1, p.data(), out.bit_count - in.bit_count);
        }
        else // Write bit as normal
        {
            uint32_t bit = 0;
            input_offset = write_bits(bit, 1, out.bytes.data(), input_offset);
        }
    }

    return out;
}

BitBuffer HammingCode::decode(const BitBuffer& in) const
{
}

int main(int argc, char** argv)
{
    struct Options
    {
        bool useSTDIN;
        enum class ECC { Repetition, Hamming };
    } options;

    for(size_t i = 0; i < argc; ++i)
    {
        if(argv[i] == "-i" || argv[i] == "--stdin")
        {
            options.useSTDIN = true;
            printf("Using stdin...\n");
        }

        // if(argv[i] == "--ecc="
    }

    RepetitionCode codec;

    const BitBuffer input{Bytes{0b00000101}, 3}; // The bit sequence 1, 0, 1.
    printf("Input bits: %s\n", input.toString().c_str());

    BitBuffer encoded = codec.encode(input); // we should obtain 111000111
    printf("Encoded bits: %s\n", encoded.toString().c_str());
    
    // Here I'll just flip one of the encoded bits
    encoded.bytes[0] ^= 0b010;
    encoded.bytes[2] ^= 0b010;
    encoded.bytes[4] ^= 0b010;

    printf("Encoded bits (corrupted): %s\n", encoded.toString().c_str());

    const BitBuffer decoded = codec.decode(encoded);
    printf("Decoded bits: %s\n", decoded.toString().c_str());
    return 0;
}

