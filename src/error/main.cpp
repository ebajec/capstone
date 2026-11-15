#include "codec.h"
#include "covq.h"

#include <algorithm>
#include <cstdio>
#include <string_view>
#include <vector>

using Bytes = std::vector<std::uint8_t>;

struct BitBuffer
{
    Bytes bytes;
    size_t bit_count = 0;

    std::string toString() const;
    static std::vector<BitBuffer> chunkBitBuffer(const BitBuffer& in, size_t size);
};

std::string BitBuffer::toString() const
{
    std::string result;
    result.reserve(bit_count);
   
    for (size_t i = 0; i < bit_count; ++i)
    {
        uint32_t bit = 0;
        read_bits(bit, 1, bytes.data(), i);
        result += std::to_string(bit);
    }

    return result;
}

std::vector<BitBuffer> BitBuffer::chunkBitBuffer(const BitBuffer& in, size_t size)
{
    std::vector<BitBuffer> out;

    for(size_t i = 0; i < in.bit_count; i += size)
    {
        const size_t min_size = std::min(size, in.bit_count - i);

        uint32_t word;
        read_bits(word, min_size, in.bytes.data(), i);

        BitBuffer component = BitBuffer{
            .bytes = Bytes((size + 7) / 8, 0),
            .bit_count = size,
        };

        write_bits(word, min_size, component.bytes.data(), 0);

        out.emplace_back(std::move(component));
    }

    return out;
}

struct IErrorCorrectingCode
{
    virtual ~IErrorCorrectingCode() = default;

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
struct HammingCode final : public IErrorCorrectingCode
{
    BitBuffer encode(const BitBuffer& in) const override;
    BitBuffer decode(const BitBuffer& in) const override;

private:
    static constexpr size_t DataBits = 8; // d
    static constexpr size_t EncodedBits = 12; // d + p

    static uint32_t encodeBlock(uint32_t data);
    static uint32_t decodeBlock(uint32_t encoded);
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
            uint32_t bit;
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
    // Shorten the final codeword to preserve the input bit count.
    const size_t d = in.bit_count % DataBits;
    size_t p = 0;
    while((1 << p) < d + p + 1)
        p++;

    BitBuffer out;
    out.bit_count = (in.bit_count / DataBits) * EncodedBits + d + p;
    out.bytes.resize((out.bit_count + 7) / 8, 0);

    size_t output_offset = 0;
    for(size_t i = 0; i < in.bit_count; i += DataBits)
    {
        const size_t min_size = std::min(DataBits, in.bit_count - i);

        uint32_t data;
        read_bits(data, min_size, in.bytes.data(), i);

        const uint32_t encoded = encodeBlock(data); // Add redundancy
        const size_t encoded_size = std::min(EncodedBits, out.bit_count - output_offset);
        output_offset = write_bits(encoded, encoded_size, out.bytes.data(), output_offset);
    }

    return out;
}

BitBuffer HammingCode::decode(const BitBuffer& in) const
{
    // Count parity positions in the final shortened codeword.
    const size_t n = in.bit_count % EncodedBits;
    size_t p = 0;
    while((1 << p) <= n)
        p++;

    BitBuffer out;
    out.bit_count = (in.bit_count / EncodedBits) * DataBits + n - p;
    out.bytes.resize((out.bit_count + 7) / 8, 0);

    size_t output_offset = 0;
    for(size_t i = 0; i < in.bit_count; i += EncodedBits)
    {
        const size_t min_size = std::min(EncodedBits, in.bit_count - i);

        uint32_t encoded;
        read_bits(encoded, min_size, in.bytes.data(), i);

        const uint32_t data = decodeBlock(encoded); // Remove redundancy and correct errors
        const size_t data_size = std::min(DataBits, out.bit_count - output_offset);
        if(data_size != 0)
            output_offset = write_bits(data, data_size, out.bytes.data(), output_offset);
    }

    return out;
}

uint32_t HammingCode::encodeBlock(uint32_t data)
{
    uint32_t out = 0;

    size_t output_offset = 0;
    for(size_t d_idx = 0; d_idx < DataBits; )
    {
        auto output_location = output_offset + 1;
        if((output_location & (output_location - 1)) == 0) // Is power of two; skip parity bit
        {
            output_offset++;
        }
        else // Write bit as normal
        {
            const uint32_t bit = (data >> d_idx) & 1;
            out |= bit << output_offset;
            output_offset++;
            d_idx++;
        }
    }

    for(size_t parity_pos = 1; parity_pos <= EncodedBits; parity_pos <<= 1)
    {
        uint32_t parity_bit = 0;

        for(size_t data_pos = 1; data_pos <= EncodedBits; ++data_pos)
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

uint32_t HammingCode::decodeBlock(uint32_t encoded)
{
    // Check parity groups
    size_t error_position = 0;
    for(size_t parity_pos = 1; parity_pos <= EncodedBits; parity_pos <<= 1)
    {
        uint32_t parity_bit = 0;

        for(size_t data_pos = 1; data_pos <= EncodedBits; ++data_pos)
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
    for(size_t i = 0; i < EncodedBits; ++i )
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

int main(int argc, char** argv)
{
    struct Options
    {
        bool useSTDIN = false;
    } options;

    for(int i = 1; i < argc; ++i)
    {
        const std::string_view& arg = argv[i];
        if(arg == "-i" || arg == "--stdin")
        {
            options.useSTDIN = true;
            printf("Using stdin...\n");
        }
    }

    HammingCode codec;

    const BitBuffer input = BitBuffer{
        .bytes = Bytes{0b1100101, 0b01101101},
        .bit_count = 16,
    }; 
    
    TIMER_BEGIN(HammingCode)
    printf("Input bits: %s\n", input.toString().c_str());

    BitBuffer encoded = codec.encode(input);
    printf("Encoded bits: %s\n", encoded.toString().c_str());
        
    // Here I'll just flip one of the encoded bits
    encoded.bytes[0] ^= 0b010;

    printf("Encoded bits (corrupted): %s\n", encoded.toString().c_str());

    const BitBuffer decoded = codec.decode(encoded);
    printf("Decoded bits: %s\n", decoded.toString().c_str());
    TIMER_END(HammingCode)

    return 0;
}

