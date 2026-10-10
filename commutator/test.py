from generate_commutators import (elementary_operators,
                                  elementary_definitions,
                                  commutator_blocks)

defs = elementary_definitions()
for op in elementary_operators():
    print(f"{op} = {defs[op]}")

blocks = commutator_blocks()
for order, block_list in blocks.items():
    print(order)
    for block in block_list:
        print("   ", block)
