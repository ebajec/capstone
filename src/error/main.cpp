#include "codec.h"
#include "timers.h"

#include <random>
#include <cstring>

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
struct RepetitionCode final : public IErrorCorrectingCode
{
    static constexpr uint32_t repetitions = 3;

    // For repetition codes, the number of encoded and decoded bits its scaled by num_repetitions
    constexpr size_t num_encoded_bits(size_t count) const override { return count * repetitions; }
    constexpr size_t num_decoded_bits(size_t count) const override { return count / repetitions; }
    constexpr const char* name() const override { return "Repetition"; }

    void encode(uint8_t* out, size_t count, const uint8_t* in) const override;
    void decode(uint8_t* out, size_t count, const uint8_t* in) const override;
};
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Hamming code: https://en.wikipedia.org/wiki/Hamming_code
// Needs data and parity bits satisfying: 2^p >= d + p + 1
// Parity bits are placed in positions which are powers of 2 (i.e. 1, 2, 4, 8, etc.)
// Parity bits are used to verify a set of bits within the code and are corrected during decoding
struct HammingCode final : public IErrorCorrectingCode
{
    static constexpr size_t DataBits = 8;
    static constexpr size_t EncodedBits = 12;

    constexpr size_t num_encoded_bits(size_t count) const override;
    constexpr size_t num_decoded_bits(size_t count) const override;
    constexpr const char* name() const override { return "Hamming"; }

    void encode(uint8_t* out, size_t count, const uint8_t* in) const override;
    void decode(uint8_t* out, size_t count, const uint8_t* in) const override;

private:
    static constexpr uint32_t encodeBlock(uint32_t data);
    static constexpr uint32_t decodeBlock(uint32_t encoded);
};
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// BCH code: 
// struct BCHCode final : public IErrorCorrectingCode
// {
//     static constexpr size_t q = 2;
//     static constexpr size_t 
//     static constexpr size_t DataBits = 8;
//     static constexpr size_t EncodedBits = 12;
//
//     constexpr size_t num_encoded_bits(size_t count) const override;
//     constexpr size_t num_decoded_bits(size_t count) const override;
//     constexpr const char* name() const override { return "BCH"; }
//
//     void encode(uint8_t* out, size_t count, const uint8_t* in) const override;
//     void decode(uint8_t* out, size_t count, const uint8_t* in) const override;
//
// private:
//     static constexpr uint32_t encodeBlock(uint32_t data);
//     static constexpr uint32_t decodeBlock(uint32_t encoded);
// };
// ---------------------------------------------------------------------------------

void RepetitionCode::encode(uint8_t* out, size_t count, const uint8_t* in) const
{
    size_t output_offset = 0;
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t bit = 0;
        read_bits(bit, 1, in, i); // Read one bit at a time

        for (uint32_t j = 0; j < repetitions; ++j)
            output_offset = write_bits(bit, 1, out, output_offset);
    }
}

void RepetitionCode::decode(uint8_t* out, size_t count, const uint8_t* in) const
{
    const size_t out_count = num_decoded_bits(count);
    size_t input_offset = 0;
    for (size_t i = 0; i < out_count; ++i)
    {
        uint32_t ones = 0;
        for (uint32_t j = 0; j < repetitions; ++j)
        {
            uint32_t bit;
            input_offset = read_bits(bit, 1, in, input_offset);
            ones += bit; // count number of ones
        }

        // If we have more ones than the number of repetitions over 2, then we decode a 1, otherwise 0
        const uint32_t bit = ones > repetitions / 2 ? 1 : 0;
        write_bits(bit, 1, out, i);
    }
}

constexpr size_t HammingCode::num_encoded_bits(size_t count) const
{
    // Shorten the final codeword to preserve the input bit count.
    const size_t d = count % DataBits;
    size_t p = 0;
    while((1 << p) < d + p + 1)
        p++;

    return (count / DataBits) * EncodedBits + d + p;
}

constexpr size_t HammingCode::num_decoded_bits(size_t count) const
{
    // Count parity positions in the final shortened codeword.
    const size_t n = count % EncodedBits;
    size_t p = 0;
    while((1 << p) <= n)
        p++;

    return (count / EncodedBits) * DataBits + (n - p);
}

void HammingCode::encode(uint8_t* out, size_t count, const uint8_t* in)  const
{
    const size_t out_count = num_encoded_bits(count);
    size_t output_offset = 0;

    for(size_t i = 0; i < count; i += DataBits)
    {
        const size_t min_size = std::min(DataBits, count - i);

        uint32_t data;
        read_bits(data, min_size, in, i);

        const uint32_t encoded = encodeBlock(data); // Add redundancy
        const size_t encoded_size = std::min(EncodedBits, out_count - output_offset);
        output_offset = write_bits(encoded, encoded_size, out, output_offset);
    }
}

void HammingCode::decode(uint8_t* out, size_t count, const uint8_t* in)  const
{
    const size_t out_count = num_decoded_bits(count);
    size_t output_offset = 0;

    for(size_t i = 0; i < count; i += EncodedBits)
    {
        const size_t min_size = std::min(EncodedBits, count - i);

        uint32_t data;
        read_bits(data, min_size, in, i);

        const uint32_t decoded = decodeBlock(data); // Remove redundancy and correct errors
        const size_t data_size = std::min(DataBits, out_count - output_offset);
        if(data_size != 0)
            output_offset = write_bits(decoded, data_size, out, output_offset);
    }
}

constexpr uint32_t HammingCode::encodeBlock(uint32_t data)
{
    uint32_t out = 0;

    size_t d_idx = 0;
    for(size_t output_offset = 0; output_offset < EncodedBits; output_offset++)
    {
        const size_t output_location = output_offset + 1;
        if((output_location & (output_location - 1)) != 0) // Is power of two; skip parity bit
        {
            const uint32_t bit = (data >> d_idx) & 1;
            out |= bit << output_offset;
            d_idx++;
        }
    }

    for(size_t parity_pos = 1; parity_pos <= EncodedBits; parity_pos <<= 1)
    {
        uint32_t parity_bit = 0;

        for(size_t data_pos = parity_pos; data_pos <= EncodedBits; ++data_pos)
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

constexpr uint32_t HammingCode::decodeBlock(uint32_t encoded)
{
    // Check parity groups
    size_t error_position = 0;
    for(size_t parity_pos = 1; parity_pos <= EncodedBits; parity_pos <<= 1)
    {
        uint32_t parity_bit = 0;

        for(size_t data_pos = parity_pos; data_pos <= EncodedBits; ++data_pos)
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

    static constexpr HammingCode hamming;
    static constexpr RepetitionCode repetition;
    constexpr std::array<const IErrorCorrectingCode*, 2> eccs = {&hamming, &repetition};

    constexpr size_t input_byte_count = 1024 * 1024; // 1 MB example
    constexpr size_t input_bit_count = input_byte_count * 8;

    auto bytes = std::make_unique<uint8_t[]>(input_byte_count);

    std::mt19937 random(0);
    for (size_t i = 0; i < input_byte_count; ++i)
        bytes[i] = uint8_t(random());

    printf("Input: %zu bytes (%zu bits)\n", input_byte_count, input_bit_count);

    bool all_match = true;
    for(const IErrorCorrectingCode* codec : eccs)
    {
        const size_t encoded_bit_count = codec->num_encoded_bits(input_bit_count);
        const size_t encoded_byte_count = codec->num_encoded_bytes(input_bit_count);
        const size_t decoded_byte_count = codec->num_decoded_bytes(encoded_bit_count);
        const size_t encoded_bits_per_input_byte = codec->num_encoded_bits(8);

        auto encoded = std::make_unique<uint8_t[]>(encoded_byte_count);
        auto decoded = std::make_unique<uint8_t[]>(decoded_byte_count);

        printf("\n%s\n", codec->name());
        TIMER_BEGIN(Encode)
        codec->encode(encoded.get(), input_bit_count, bytes.get());
        TIMER_END(Encode)

        printf("Encoded: %zu bytes (%zu bits)\n", encoded_byte_count, encoded_bit_count);

        for(size_t i = 2; i < encoded_bit_count; i += encoded_bits_per_input_byte)
        {
            encoded[i / 8] ^= uint8_t(1u << (i % 8));
        }

        TIMER_BEGIN(Decode)
        codec->decode(decoded.get(), encoded_bit_count, encoded.get());
        TIMER_END(Decode)

        const bool matches = decoded_byte_count == input_byte_count &&
            std::memcmp(decoded.get(), bytes.get(), input_byte_count) == 0;
        printf("Decoded matches input: %s\n", matches ? "yes" : "no");
        all_match &= matches;
    }

    return all_match ? 0 : 1;
}

