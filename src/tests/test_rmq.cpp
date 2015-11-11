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
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>

#include "rmq.hpp"

template class std::vector<int>;

class rmq_tester
{
    std::size_t size;
    std::vector<int> els;
    rmq<std::vector<int>::iterator>* minquery;
public:
    rmq_tester(std::size_t size) : size(size), els(size)
    {
        std::generate(els.begin(), els.end(), [](){return std::rand();});
        minquery = new rmq<std::vector<int>::iterator>(els.begin(), els.end(), 8, 4);
    }

    bool test(int start, int end)
    {
        std::vector<int>::iterator min_it = minquery->query(els.begin()+start, els.begin()+end);
        if (*min_it != *std::min_element(els.begin()+start, els.begin()+end))
        {
            std::cerr << "ERROR: range(" << start << "," << end << ")=" << *(min_it) << " at i=" << min_it - els.begin() << std::endl;
            return false;
        }
        else
            return true;
    }

    void test_all()
    {
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = i+1; j < size; ++j)
            {
                if (!test(i,j))
                {
                    std::cerr << "Error with range(" << i << "," << j << ")" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
    }

    ~rmq_tester()
    {
        delete minquery;
    }
};

void timing_comp(std::size_t size)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int> els(size);
    std::generate(els.begin(), els.end(), [](){return std::rand() % 1000;});

    auto now = std::chrono::high_resolution_clock::now();
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    start_time = now;
    std::cerr << "Generate input: " << duration_ms << std::endl;

    rmq<std::vector<int>::iterator> rmq(els.begin(), els.end());

    now = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    std::cerr << "Construction RMQ: " << duration_ms << std::endl;

    int rand_tests = size;
    double rmq_queries = 0.0;
    double minel_queries = 0.0;
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rand_tests; ++i)
    {
        // get random range and start timing
        int range_start = std::rand() % (size - 2);
        int range_end = std::rand() % (size - range_start - 1) + range_start;

        // query RMQ
        auto min = rmq.query(els.begin()+range_start, els.begin()+range_end);

        // stop timing
        //now = std::chrono::high_resolution_clock::now();
        //duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        //rmq_queries += duration_ms;
        //start_time = now;

        // use linear std::min_element
        auto min2 = std::min_element(els.begin()+range_start, els.begin() + range_end);

        // stop time
        //now = std::chrono::high_resolution_clock::now();
        //duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        //minel_queries += duration_ms;

        if (*min != *min2)
        {
            std::cerr << "ERROR: different mins for range(" << range_start << "," << range_end << ")" << std::endl;
            std::cerr << "rmq at " << min - els.begin() << " = " << *min << std::endl;
            std::cerr << "mne at " << min2 - els.begin() << " = " << *min2 << std::endl;
            exit(EXIT_FAILURE);
        }
        else
            std::cerr << "." << std::endl;
    }

    now = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    std::cout << "RMQ              queries total time: " << duration_ms/1000.0 << "s" << std::endl;
    //std::cout << "std::min_element queries total time: " << minel_queries/1000.0 << "s" << std::endl;
}

int main(int argc, char *argv[])
{
    //timing_comp(1000000037);
    /*
    for (int size = 1; size < 1000; ++size)
    {
        rmq_tester rmq(size);
        std::cerr << "Testing with size: " << size << std::endl;
        rmq.test_all();
    }
    */
    //rmq_tester rmq(10000);
    //rmq.test_all();
    timing_comp(100000);



    return 0;
}
