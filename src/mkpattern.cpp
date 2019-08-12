#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

std::string read_file(const std::string& filename) {
    // read input file into in-memory string
    std::string input_str;
    std::ifstream t(filename.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    input_str = buffer.str();
    return input_str;
}

std::vector<std::string> rand_patterns(const std::string& str, size_t len, size_t num) {
    std::vector<std::string> patterns;
    patterns.reserve(num);
    std::minstd_rand g;
    std::uniform_int_distribution<size_t> d(0, str.size()-len-1);
    for (size_t i = 0; i < num; ++i) {
        size_t start = d(g);
        patterns.emplace_back(str.begin() + start, str.begin() + start + len);
    }
    return patterns;
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage:  ./mkpattern <filename> <num> <len>" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string filename(argv[1]);
    int num = std::stoi(argv[2]);
    int len = std::stoi(argv[3]);

    if (num <= 0 || len < 1) {
        std::cerr << "Usage:  ./mkpattern <filename> <num> <len>" << std::endl;
        exit(EXIT_FAILURE);
    }

    // read input file
    std::string input = read_file(filename);

    // generate random patterns of given length
    std::vector<std::string> patterns = rand_patterns(input, len, num);

    // output patterns in text format
    for (const std::string& P : patterns) {
        std::cout << P << std::endl;
    }

    return 0;
}
