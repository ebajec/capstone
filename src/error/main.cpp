#include "ecc.h"
#include "timers.h"

#include <random>

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
    struct BenchmarkCase
    {
        const IErrorCorrectingCode* codec;
        size_t error_stride;
    };
    constexpr std::array eccs = {
        BenchmarkCase{&hamming, hamming.OutputBytes * 8 / hamming.CodewordsPerBlock},
        BenchmarkCase{&repetition, repetition.num_encoded_bits(8)},
    };

    constexpr size_t input_byte_count = 1024 * 1024; // 1 MB example
    constexpr size_t input_bit_count = input_byte_count * 8;

    auto bytes = std::make_unique<uint8_t[]>(input_byte_count);

    std::mt19937 random(0);
    for (size_t i = 0; i < input_byte_count; ++i)
        bytes[i] = uint8_t(random());

    printf("Input: %zu bytes (%zu bits)\n", input_byte_count, input_bit_count);

    bool all_match = true;
    for(;;)
    {
        for(const auto& [codec, error_stride] : eccs)
        {
            const size_t encoded_bit_count = codec->num_encoded_bits(input_bit_count);
            const size_t encoded_byte_count = codec->num_encoded_bytes(input_bit_count);
            const size_t decoded_byte_count = codec->num_decoded_bytes(encoded_bit_count);

            auto encoded = std::make_unique<uint8_t[]>(encoded_byte_count);
            auto decoded = std::make_unique<uint8_t[]>(decoded_byte_count);

            printf("\n%s\n", codec->name());
            TIMER_BEGIN(Encode)
            codec->encode(encoded.get(), input_bit_count, bytes.get());
            TIMER_END(Encode)

            printf("Encoded: %zu bytes (%zu bits)\n", encoded_byte_count, encoded_bit_count);

            for(size_t i = 2; i < encoded_bit_count; i += error_stride)
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
    }

    return all_match ? 0 : 1;
}

