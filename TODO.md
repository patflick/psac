- [x] integrate libdivsufsort as reference (for single node run time and
      testing)
- [~] unified C++ interface between libdivsufsort and psac
- [x] (see previous) psac interface for global mem (scatter + psac + gather)
- [x] implement testing against libdivsufsort
- [x] test against chr22
- [ ] no 3-tuples in samplesort -> use only initial bucket and then switch
      to bucket chaising
- [ ] document library functions
- [ ] clean iterator interface
- [ ] distributed correctness test
- [ ] more fine grained performance profiling
- [ ] multiseq merge after samplesort all2all (should be faster than another
      round of sorting) [in-place]
- [ ] find other bottlenecks and optimize
- [ ] LCP
- [ ] BWT
- [ ] get wavefront and other paper's code and compare
- [ ] implementation for few unresolved buckets
- [ ] clean up test code
