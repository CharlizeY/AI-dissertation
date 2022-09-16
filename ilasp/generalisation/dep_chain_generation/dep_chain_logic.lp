dep_chain(L1, L2) :- dep(L1, X, Y), dep(L2, Y, Z).

#show dep_chain/2.
