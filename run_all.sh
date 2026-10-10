#!/usr/bin/env bash
# Run the full derivation pipeline with all numerical checks.
# Stage 5 (cumulant_decomposition/spin_free) is an input formula, not a
# program; it is verified inside stage 6.
set -e
cd "$(dirname "$0")"

echo "=== 1. generalized Wick theorem (self test) ==="
(cd generalized_wick_theorem && python3 generalized_wick_commutator.py)
echo
echo "=== 2. CT commutators ==="
(cd commutator && python3 generate_commutators.py)
echo
echo "=== 3. many-body expansion ==="
(cd many_body_equation && python3 expand_commutators.py)
echo
echo "=== 4. space restriction ==="
(cd space_restriction && python3 space_restriction.py)
echo
echo "=== 6. cumulant expansion (also verifies the stage-5 formula) ==="
(cd cumulant_expansion && python3 cumulant_expansion.py)
echo
echo "All stages passed."
