import torch

ROWS = 8
COLS = 64
VEC = 8
K = 8

def S(r):
  return VEC * (r % K)

A = torch.randint(0, 10, (ROWS, COLS)).tolist()
print("Before swizzling:")
for row in A:
  for j in range(VEC):
    row[j] = j
  print(row)
print()

# Perform swizzling
swizzled_A = []
for i in range(ROWS):
  row = [0] * COLS
  print(f"S({i}) = {S(i)}")
  for j in range(COLS):
    idx = j ^ S(i)
    row[idx] = A[i][j]
  swizzled_A.append(row)

print("\nAfter swizzling:")
for row in swizzled_A:
  print(row)
print()
