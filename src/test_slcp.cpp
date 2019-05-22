

#include <mxx/env.hpp>
#include <mxx/comm.hpp>

#include <mxx/distribution.hpp>

#include <iostream>
#include <vector>
#include <tldt.hpp>
#include <seq_query.hpp>

#include <cxx-prettyprint/prettyprint.hpp>

int main(int argc, char *argv[])
{
    mxx::env e(argc, argv);
    mxx::comm c;

    size_t n = 1000;
    size_t maxsize = 20;

    salcp_index<size_t> idx;
    if (c.rank() == 0) {
        std::string s = rand_dna(n, 13);
        //std::cout << "s = " << s << std::endl;
        idx.construct(s.begin(), s.end());
    }
    std::vector<size_t> local_LCP = stable_distribute(idx.LCP, c);


    std::vector<size_t> off = sample_lcp_distr(local_LCP, maxsize, c);


    std::vector<size_t> goff = mxx::gatherv(off, 0, c);

    if (c.rank() == 0) {
        seq_check_sample(idx.LCP, goff, maxsize);
    }


    return 0;
}
