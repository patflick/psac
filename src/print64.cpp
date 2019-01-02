#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: ./print64 <filename>" << std::endl;
        exit(EXIT_FAILURE);
    }


    FILE * f = fopen(argv[1], "rb"); 
    if (f == NULL) {
        std::cerr << "Error opening file `" << argv[1] << "`." << std::endl;
        exit(EXIT_FAILURE);
    }

    // get file size
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);

    if (size % 8 != 0) {
        std::cerr << "Error: file `" << argv[1] << "` has size which is not a multiple of 64bits." << std::endl;
        exit(EXIT_FAILURE);
    }

    // read file
    size_t count = size/sizeof(uint64_t);
    std::vector<uint64_t> buf(count);
    size_t read_count = fread(&buf[0], sizeof(uint64_t), count, f);
    if (read_count != count) {
        std::cerr << "Unexpected error reading file." << std::endl;
        exit(EXIT_FAILURE);
    }

    // output file in decimal format
    for (size_t i = 0; i < buf.size(); ++i) {
        std::cout << buf[i] << std::endl;
    }
    return 0;
}
