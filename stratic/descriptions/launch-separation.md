# Optional launch-level separation

Output addresses can be required to differ from every weight, input and bias
address and from each other. `OutputsSeparated` expresses precisely those
four families of disequalities in the modular word-address model. Read-only
input addresses may overlap when their specified values agree.

This condition is separate from the single-thread theorem. That theorem reads
all inputs before its only store and preserves every address other than its
selected output address, including when that address aliases an input.
Separation is retained as an optional contract for future composition of
different threads; no all-thread or GPU-interleaving theorem is claimed here.
