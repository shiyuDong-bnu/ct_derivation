In this section we need to expand all commutator out using
generalized wick theorems.

the all comutator is in comutator directory
the generalized theorem is in generalized wick theorem

simplify rule 1: resolve_delta:
To simplify the equation we should note that 
p,q,r,s represent the whole space
i,j,k,l represent occ space
a,b,c,d represent vir space 
and \delta (whole space, vir space) should left with vir space
and \delta (whole space, occ space) should left with occ space
and \delta (occ  space , vir space) should be zero

simplify rule 2: canonicalization
 (orbital space)the operator have the symmetry that E^{p_1,p_2,\cdots,p_n}_{q_1,q_2,\cdots,q_n}we can permute the upper and lower index simutaneously just keep the p_i and q_i paired.to canonicalize this we just need to specify the upper index order. the upper index order is sorted first by the space and then by lexical order.
simplify rule 3: dummy index 
after we sort operator accoring to space. we substitude the operator index to lecial order
for example E^{rsb}_{ija}->E^{p_1 p_2 a_1}_{i_1 i_2 a_2} we should substitude the tensor part too to keep this consistent.

simplify rule 4: tensor index 
tensor have those two symmetry
p1. the pair symmetry of the amplitudes, t^{ab}_{ij} = t^{ba}_{ji} and
   g^{pq}_{rs} = g^{qp}_{sr} (simultaneous swap of both index pairs) --
   no index antisymmetry of t or g is used anywhere;
p2. real coefficients: f symmetric (f^q_p = f^p_q) and g symmetric under
   pair exchange (g^{pq}_{rs} = g^{rs}_{pq}); the density matrices D
   (appearing in the cumulant stage) carry the same symmetries;
so you do permutation to generate the orbital by the symmetry group and then sort them 
according to the orbital space
the priority is upper 1>upper2 \cdots >lower1 lower2 etc .
first sort group and then index.
simplify rule 5: dummy index 
like rule3 we substitue dummy index accooring to the number

!! do not use fermionic property which permutate two upper index and do not permutate lower index we are using spin summationed operator which do not have this property.
!! do not use sympy default canonical function which may break the rule latentlyl.

after this map {p1,p2,pn}->{p,q,r,s,t,u}etc

we will do cumulant expansion of three body operator later.
OK,to simplify the result furthur we could canonicalize the result