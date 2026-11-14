#include "codec.h"
#include <ev2/core.h>

#include <filesystem>

using Bytes = std::vector<std::uint8_t>;

struct BitBuffer
{
    Bytes bytes;
    size_t bit_count = 0;
};

struct ErrorCorrectingCode
{
    
};

// Basic repitition code as a simplicitic ECC
struct RepetititionCode : public ErrorCorrectingCode
{
    uint32_t* bits;

    BitBuffer encode(const BitBuffer& in) const;
    BitBuffer decode(const BitBuffer& in) const;
};


namespace Utility
{
    Bytes read_file(const std::filesystem::path& path);
    void write_file(const std::filesystem::path& path, const Bytes& bytes);
    void corrupt_bits(BitBuffer& data, float epsilon);
}

Bytes Utility::read_file(const std::filesystem::path& path);
void Utility::write_file(const std::filesystem::path& path, const Bytes& bytes);
void Utility::corrupt_bits(BitBuffer& data, float epsilon);


int main(int argc, char** argv)
{
    printf("error something something\n");

    return 0;
}

