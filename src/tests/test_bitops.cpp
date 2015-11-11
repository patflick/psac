/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
