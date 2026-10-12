# Execution context and return

Every instruction execution receives a required read-only context containing
its immutable thread identity, parameter/conversion environment and arithmetic
operations. There is no absent-arithmetic mode. Integer-only fragments work
for arbitrary supplied arithmetic because they do not invoke it.

Instructions read operands before updating destinations. `writeNext` and
`write64Next` explicitly write one register and advance the PC by one. Branches
select their own next PC. Missing instruction fetches, input words, parameter
slots or address conversions produce no modeled successor.

`Returns program ctx initial final` means there exists a finite instruction
count for which execution succeeds with that final state, and the final state's
returned flag is true. Running zero instructions from an active state does not
establish return. A returned state has no next machine step. Exact instruction
counts remain available to the block-composition proofs and execution checks.

The selected layer context uses its configured thread identity and concrete
FloatLib binary32 arithmetic. This relation specifies execution in the formal
machine; device scheduling and hardware execution are outside its claim.
