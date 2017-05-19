#include <mxx/env.hpp>
#include <mxx/comm.hpp>

#include "bitops.hpp"
#include "lcp.hpp"

#define SDEBUG(x) mxx::sync_cerr(c) << "[" << c.rank() << "]: " #x " = " << (x) << std::endl
#define DEBUG(x) std::cout << "" #x " = " << (x) << std::endl;

int main(int argc, char *argv[]) {
    mxx::env(argc, argv);

    unsigned int x = 8;
    unsigned int y = 8;
    unsigned int l = lcp_bitwise_no0(8, 8, 2, 2);

    unsigned int z = x ^ y;
    unsigned int lz = leading_zeros(z);
    // if x==y, then return the trailing zeroes of both combined, else 0
    unsigned int tz = trailing_zeros((!(x == y)) | (x | y));
    DEBUG(l);
    DEBUG(lz);
    DEBUG(tz);

    return 0;
}
