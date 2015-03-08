#include <iostream>
#include <cstdlib>
#include <cstdint>
#include "bitops.hpp"


// TODO do these things as google test
int main()
{
    // run a large set of random tests
    unsigned int sum = 0;
    int num_tests = 100000000;
    for (int i = 0; i < num_tests; ++i)
    {
        uint64_t x = 0;
        x |= static_cast<uint64_t>(rand()) << 32;
        x |= static_cast<uint64_t>(rand());

        unsigned int t1 = trailing_zeros(x);
        unsigned int t2 = reference_trailing_zeros(x);
        //unsigned int t2 = t1;
        //sum += t2;

        unsigned int l1 = leading_zeros(x);
        unsigned int l2 = reference_leading_zeros(x);
        //unsigned int l2 = l1;
        //sum += l2;

        if (t1 != t2) {
            std::cerr << "Error! trailing zeros: " << t1 << " != " << t2 << std::endl;
        }
        if (l1 != l2) {
            std::cerr << "Error! leading zeros: " << l1 << " != " << l2 << std::endl;
        }
    }
    return sum;
}
