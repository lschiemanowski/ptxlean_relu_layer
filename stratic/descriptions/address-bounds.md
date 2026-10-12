# Optional natural-address bounds

The address-calculation block also admits a stronger interpretation under
explicit nonwrapping buffer-end bounds. The row lies below the output count;
the input width is positive and its four-byte row stride fits a 32-bit immediate.
Each base plus its complete weight, bias or output buffer size is strictly
below `2^64`.

Then the computed weight-row, bias-element and output-element pointers equal
the corresponding natural-number addresses, and each admits a full four-byte
element inside its buffer. The concrete layer specializes these bounds to the
shared dimensions and configured launch.

These are optional arithmetic bounds. The main kernel proof uses modular word
addresses and does not depend on this module. Allocation, initialized contents,
access permissions, alignment and CUDA device correspondence are not proved
by these address inequalities.
