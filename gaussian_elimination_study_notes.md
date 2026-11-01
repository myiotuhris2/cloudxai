# Gaussian Elimination and Recursion: Code and Step-by-Step Explanations

This file collects every question and answer from the conversation (Gaussian elimination code, NumPy details, back substitution, polynomial multiplication, and the recursive `solve_equations`), followed by your own notes as an appendix.

**Contents**

1. Gaussian elimination: full code with explanations
2. Q: What do `~` and `isclose` do in `check_solution`?
3. Q: How does `back_substitution` work (in detail)?
4. Visual: back substitution trace
5. Q: What is `if pivot != i: M = swap_rows(M, i, pivot)`?
6. Q: What is `axis=1` in `np.all(..., axis=1)`?
7. Visual: `multiply_polynomials` step by step
8. Visual: recursive `solve_equations` step by step
9. Appendix: your notes

---

## 1. Gaussian elimination: full code with explanations

The notebook (C1W2 assignment) solves a system of linear equations by turning the augmented matrix `[A | B]` into row echelon form and then solving from the bottom up.

Example system used in the notebook:

```
 2x1 + 3x2 + 5x3 = 12
-3x1 - 2x2 + 4x3 = -2
  x1 +  x2 - 2x3 = 8
```

Only the intro cells were visible, so the function names and signatures in the graded cells may differ. Match them to what the notebook asks for.

### Step 1: Augmented matrix

```python
import numpy as np

def augmented_matrix(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)  # make B a column vector
    return np.hstack((A, B))                     # glue B onto the right of A
```

`hstack` places B beside A, giving `[A | B]`. Using `float` avoids integer-division surprises later.

### Step 2: The three row operations

```python
def swap_rows(M, i, j):
    M = M.copy()
    M[[i, j]] = M[[j, i]]          # Row Switching
    return M

def multiply_row(M, i, k):
    assert k != 0, "Scalar must be non-zero"
    M = M.copy()
    M[i] = M[i] * k                # Row Scaling
    return M

def add_rows(M, src, dst, k):
    M = M.copy()
    M[dst] = M[dst] + k * M[src]   # Row Replacement
    return M
```

Each function copies the matrix first, so the original is never modified by accident.

### Step 2 (continued): Reduce to row echelon form

```python
def row_echelon_form(M):
    M = M.copy().astype(float)
    n = M.shape[0]
    for i in range(n):
        # Pick the row (from i downward) with the largest value in column i
        pivot = i + np.argmax(np.abs(M[i:, i]))
        if np.isclose(M[pivot, i], 0):
            continue                           # no pivot in this column
        if pivot != i:
            M = swap_rows(M, i, pivot)         # move pivot to the diagonal
        for j in range(i + 1, n):              # zero out everything below
            factor = -M[j, i] / M[i, i]
            M = add_rows(M, i, j, factor)
    return M
```

For each column, find a pivot, swap it onto the diagonal, then add multiples of the pivot row to the rows below to make zeros. Choosing the largest pivot also improves numerical stability.

### Step 3: Solution check

```python
def check_solution(M):
    n = M.shape[0]
    A_part = M[:, :n]
    b_part = M[:, -1]
    zero_rows = np.all(np.isclose(A_part, 0), axis=1)  # rows where A-part is all 0

    if not zero_rows.any():
        return "unique"                                 # case 1
    if np.any(~np.isclose(b_part[zero_rows], 0)):
        return "none"                                   # case 2: 0 = nonzero
    return "infinite"                                   # case 3: only 0 = 0 rows
```

This implements the three cases from the notebook:

| Case | What you see in the matrix | Result |
|---|---|---|
| 1 | No row of zeros in the square part | unique solution |
| 2 | A zero row with a non-zero augmented part (`0 = 5`) | no solution |
| 3 | Zero rows whose augmented part is also 0 (`0 = 0`) | infinitely many solutions |

Case 2 takes priority, because one contradictory row makes the system impossible even if other rows are `0 = 0`.

### Step 5: Back substitution

```python
def back_substitution(M):
    n = M.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                     # last row upward
        x[i] = (M[i, -1] - M[i, i+1:n] @ x[i+1:n]) / M[i, i]
    return x
```

The last row has one unknown, so solve it directly. For each row above, subtract the already-known variables (the dot product) from the right-hand side, then divide by the diagonal entry. Sections 3 and 4 explain this in detail.

### Step 6: Put it together

```python
def gaussian_elimination(A, B):
    M = augmented_matrix(A, B)
    M = row_echelon_form(M)
    status = check_solution(M)

    if status == "unique":
        return back_substitution(M)
    elif status == "none":
        return "The system has no solution."
    else:
        return "The system has infinitely many solutions."
```

### Test with the example from the notebook

```python
A = [[2, 3, 5], [-3, -2, 4], [1, 1, -2]]
B = [12, -2, 8]
print(gaussian_elimination(A, B))   # approximately [-14.  17.27  -2.364]
```

Checked by hand: x = −14, y = 190/11, z = −26/11.

**Notes**

- The screenshots jump from Step 3 to Step 5, so Step 4 is either hidden in the notebook or not shown. Check what it asks for, because it may want a specific function.
- The notebook's autograder likely expects particular function names and return formats, so adapt the signatures to the starter code.

---

## 2. Q: What do `~` and `isclose` do in `check_solution`?

```python
if np.any(~np.isclose(b_part[zero_rows], 0)):
    return "none"
```

Read it from the inside out.

### 2.1 `b_part[zero_rows]`

`zero_rows` is a boolean array like `[False, False, True]`, one entry per row, marking rows where the whole coefficient part is zeros. Indexing with it keeps only the augmented-column (right-hand side) values of those rows.

Example: if `b_part = [12, 5, 0.0]` and `zero_rows = [False, False, True]`, then `b_part[zero_rows]` is `[0.0]`.

These are the right-hand sides of rows that look like `0x + 0y + 0z = ?`, so the question is whether `?` is 0.

### 2.2 `np.isclose(..., 0)`

This checks "is this value approximately 0?" for every element and returns booleans.

Why not `== 0`? Computers store decimals with tiny rounding errors. After elimination, a value that should be exactly 0 often ends up as `1.2e-16` or `-3e-17`, and `1.2e-16 == 0` is `False`. `isclose` treats it as zero. It tests:

```
|a - b| <= atol + rtol * |b|      (defaults: atol=1e-8, rtol=1e-5)
```

So `np.isclose(1e-16, 0)` is `True`, while `np.isclose(0.5, 0)` is `False`.

### 2.3 `~`

`~` is the element-wise NOT for boolean arrays: it flips every `True` to `False` and vice versa.

- `isclose(..., 0)` gives `True` where the value is approximately 0.
- `~` of that gives `True` where the value is **not** approximately 0.

You can't use Python's `not` here, because it raises an error on arrays with more than one element. `~` is the array version.

### 2.4 `np.any(...)`

This returns `True` if at least one element is `True`.

### 2.5 Putting it together

Is there any all-zero row whose right-hand side is non-zero? Such a row says `0 = 5`, which is impossible, so the system has **no solution**.

| Row | `b` value | `isclose(b,0)` | `~` | Meaning |
|---|---|---|---|---|
| `0 0 0 \| 0` | 0 | True | False | fine (`0 = 0`) |
| `0 0 0 \| 7` | 7 | False | True | contradiction |

If any entry is `True` after the `~`, `np.any` returns `True` and the function returns `"none"`.

---

## 3. Q: How does `back_substitution` work (in detail)?

After elimination, the matrix is upper triangular. Take this small example:

```
[ 2  1 -1 |  3 ]   row 0
[ 0  3  2 |  7 ]   row 1
[ 0  0  4 |  8 ]   row 2
```

This represents:

```
2x0 +  x1 -  x2 = 3
       3x1 + 2x2 = 7
             4x2 = 8
```

The bottom row has only one unknown, so you solve it first, then use that answer in the row above, and so on. That is back substitution.

### Line by line

```python
n = M.shape[0]
```
`shape[0]` is the number of rows, here 3, which is also the number of unknowns because A is square.

```python
x = np.zeros(n)
```
This creates the answer vector `[0., 0., 0.]`. The entries are placeholders that get filled in from the bottom up.

```python
for i in range(n - 1, -1, -1):
```
`range(start, stop, step)` with start `n-1`, stop `-1` (excluded) and step `-1` counts backwards: `2, 1, 0`. That is last row to first.

```python
x[i] = (M[i, -1] - M[i, i+1:n] @ x[i+1:n]) / M[i, i]
```

This line is the whole algorithm. The pieces:

- `M[i, -1]` is the last column of row `i`, the right-hand side b.
- `M[i, i+1:n]` holds the entries to the **right of the diagonal** in row `i`, which are the coefficients of variables already solved.
- `x[i+1:n]` holds the values already found for those variables.
- `@` is the dot product: multiply matching pairs and add them up.
- `M[i, i]` is the diagonal entry, the coefficient of the variable being solved now.

For row `i`, the equation is `M[i,i]*x_i + (known stuff) = b`. Rearranging:

```
x_i = (b - known stuff) / M[i,i]
```

That is exactly the line.

### Trace through the example

**i = 2 (last row)**
- `M[2, 3:3]` and `x[3:3]` are empty slices, so the dot product is 0.
- `x[2] = (8 - 0) / 4 = 2`
- `x = [0, 0, 2]`

**i = 1**
- `M[1, 2:3] = [2]`, `x[2:3] = [2]`, so the dot product is `2*2 = 4`.
- `x[1] = (7 - 4) / 3 = 1`
- `x = [0, 1, 2]`

**i = 0**
- `M[0, 1:3] = [1, -1]`, `x[1:3] = [1, 2]`, so the dot product is `1*1 + (-1)*2 = -1`.
- `x[0] = (3 - (-1)) / 2 = 2`
- `x = [2, 1, 2]`

Check in the original equations: `2(2)+1-2 = 3` OK, `3(1)+2(2) = 7` OK, `4(2) = 8` OK.

**Why the empty slice works for the last row:** a dot product of two empty arrays is 0, so the same formula handles every row, including the last one, with no special case.

**Why we never divide by zero here:** back substitution only runs when `check_solution` returned `"unique"`, which guarantees no zero rows. For a square system with a unique solution, every diagonal entry is non-zero.

---

## 4. Visual: back substitution trace

This is a static version of the interactive walkthrough. Legend: **D** = diagonal `M[i, i]` (what you divide by), **K** = known coefficients `M[i, i+1:n]`, **R** = right-hand side `M[i, -1]`.

Matrix:

```
          c0   c1   c2 | rhs
row 0  [   2    1   -1 |   3 ]
row 1  [   0    3    2 |   7 ]
row 2  [   0    0    4 |   8 ]
```

| Step | Row `i` | D | K | R | `x` vector after | Calculation |
|---|---|---|---|---|---|---|
| Start | n/a | n/a | n/a | n/a | `[?, ?, ?]` | `x = np.zeros(3)` placeholders |
| 1 | 2 | 4 | none (empty) | 8 | `[?, ?, 2]` | `(8 - 0) / 4 = 2` |
| 2 | 1 | 3 | `[2]` times `x[2:3]=[2]` | 7 | `[?, 1, 2]` | `(7 - 2*2) / 3 = 1` |
| 3 | 0 | 2 | `[1, -1]` times `x[1:3]=[1, 2]` | 3 | `[2, 1, 2]` | `(3 - (1*1 + (-1)*2)) / 2 = 2` |

**Row 1 as a worked example:** the equation is `3*x1 + 2*x2 = 7`. You already know `x2 = 2`, so `3*x1 + 2*2 = 7`, which gives `3*x1 = 7 - 4`, so `x1 = 3/3 = 1`. The code line does exactly this: `(7 - dot) / 3`, where `dot` is `2*2`.

---

## 5. Q: What is `if pivot != i: M = swap_rows(M, i, pivot)`?

This is the row switching step from the notebook. It moves the best pivot row up onto the diagonal.

```python
pivot = i + np.argmax(np.abs(M[i:, i]))
if np.isclose(M[pivot, i], 0):
    continue
if pivot != i:
    M = swap_rows(M, i, pivot)
```

**1. `pivot` is a row number, not a value.**
- `M[i:, i]` is column `i`, from row `i` downward.
- `np.abs(...)` takes absolute values, so -5 counts as bigger than 3.
- `np.argmax(...)` returns the *position* of the largest value within that slice. That position starts counting from 0 at row `i`, so adding `i` converts it back to a real row number in the full matrix.

**2. `if pivot != i:` asks whether the best row is already on the diagonal.**
- If `pivot == i`, the row with the largest value is already in place, so there is nothing to swap.
- If `pivot != i`, a lower row has a better pivot, so we swap it up.

Skipping the swap when it isn't needed just avoids a pointless operation.

**3. `M = swap_rows(M, i, pivot)`** swaps row `i` with row `pivot`. Remember `swap_rows` returns a new matrix, so you have to assign it back to `M`.

### Example at i = 0

```
row 0:  [ 1   1  -2 |  8 ]
row 1:  [-3  -2   4 | -2 ]
row 2:  [ 2   3   5 | 12 ]
```

Column 0 from row 0 down is `[1, -3, 2]`. The absolute values are `[1, 3, 2]`, and `argmax` gives position 1, so `pivot = 0 + 1 = 1`. Since `1 != 0`, row 1 swaps with row 0:

```
row 0:  [-3  -2   4 | -2 ]   <- new pivot is -3
row 1:  [ 1   1  -2 |  8 ]
row 2:  [ 2   3   5 | 12 ]
```

### Why swap at all?

1. **It is sometimes required.** If the diagonal entry is 0, you can't divide by it. For example, `[0 2 | 4]` above `[3 1 | 5]` needs a swap before elimination can work.
2. **It improves accuracy.** Dividing by a tiny pivot such as 0.0001 magnifies rounding errors. Picking the largest value in the column keeps the factors `-M[j, i] / M[i, i]` small. This is called partial pivoting.

Swapping rows doesn't change the solution, because it is just writing the same equations in a different order.

---

## 6. Q: What is `axis=1` in `np.all(..., axis=1)`?

```python
zero_rows = np.all(np.isclose(A_part, 0), axis=1)
```

`axis` tells NumPy which direction to collapse when it combines values. For a 2D matrix:

- `axis=0` works **down the columns** (one result per column).
- `axis=1` works **across the rows** (one result per row).

A handy way to remember it: the axis you give is the one that disappears. A 3x3 matrix has shape `(3, 3)`. With `axis=1`, the column dimension collapses and you're left with 3 results, one per row.

### Step by step on an example

Say `A_part` after elimination is:

```
[[2.  3.  5. ]
 [0.  1.  2. ]
 [0.  0.  0. ]]
```

**1. `np.isclose(A_part, 0)`** returns a True/False matrix of the same shape:

```
[[False False False]
 [ True False False]
 [ True  True  True]]
```

**2. `np.all(..., axis=1)`** asks, for each row, whether *all* entries in that row are True:

```
row 0: False, False, False  ->  False
row 1: True,  False, False  ->  False
row 2: True,  True,  True   ->  True
```

Result: `[False, False, True]`. That is the `zero_rows` array: True marks a row whose coefficient part is entirely zeros.

### What would the other options do?

- `axis=0` would check down each column instead and give `[False, False, False]` here, which is the wrong question.
- With no axis, `np.all` collapses the whole matrix to a single True/False ("is the entire matrix zero?"). That is also not what we want, because we need to know *which* rows are zero.

`np.any(..., axis=1)` follows the same idea: per row, is at least one entry True?

Once you have `zero_rows = [False, False, True]`, you can use it as a filter, like `b_part[zero_rows]`, to pull out the right-hand side values of only the all-zero rows.

---

## 7. Visual: `multiply_polynomials` step by step

Polynomials are stored as coefficient lists, **highest power first**. For example `[2, -1, 3]` means `2x^2 - x + 3`.

```python
def multiply_polynomials(p1, p2):
    result = [0]
    for i in range(len(p1)):
        c = p1[i]
        power = len(p1) - 1 - i
        scaled = multiply_polynomial(p2, c)
        shifted = scaled + [0] * power
        result = add_polynomials(result, shifted)
    return result
```

(In your notebook these helpers are named `multiply_polynomial` for the scalar multiply and `add_poly` for addition. Use the same names everywhere.)

**The idea:** this is the same as multiplying by hand. Take each term of `p1` one at a time, multiply all of `p2` by it, then add up the partial answers.

**What each line does**

- `c = p1[i]` picks the current coefficient.
- `power = len(p1) - 1 - i` is the power of x that goes with that coefficient. Index 0 has the highest power, and the last index has x^0.
- `scaled = multiply_polynomial(p2, c)` multiplies every number in `p2` by `c`.
- `shifted = scaled + [0] * power` handles the x part of the term. Adding a zero on the right of a coefficient list multiplies the polynomial by x, so `power` zeros multiply it by x^power.
- `result = add_polynomials(result, shifted)` adds the partial answer into the running total. Lists are added from the right (the lowest power).

### Trace: (2x^2 - x + 3) x (x + 2), with `p1 = [2, -1, 3]`, `p2 = [1, 2]`

| Pass `i` | `c` | `power` | `scaled` = p2 x c | `shifted` | `result` before | `result` after |
|---|---|---|---|---|---|---|
| start | n/a | n/a | n/a | n/a | n/a | `[0]` |
| 0 | 2 | 2 | `[2, 4]` | `[2, 4, 0, 0]` | `[0]` | `[2, 4, 0, 0]` |
| 1 | -1 | 1 | `[-1, -2]` | `[-1, -2, 0]` | `[2, 4, 0, 0]` | `[2, 3, -2, 0]` |
| 2 | 3 | 0 | `[3, 6]` | `[3, 6]` | `[2, 3, -2, 0]` | `[2, 3, 1, 6]` |

Aligned by power for the last pass:

```
              x^3  x^2  x^1  x^0
result before   2    3   -2    0
shifted         0    0    3    6
result after    2    3    1    6
```

Final answer: `[2, 3, 1, 6]` means `2x^3 + 3x^2 + x + 6`.

Check of the first pass by hand: `2x^2 * (x + 2) = 2x^3 + 4x^2`, which is `[2, 4, 0, 0]`.

**Notes**

- If you define two functions with the same name, the later one replaces the earlier one. In your notebook `multiply_poly` is first defined as a scalar multiply and later redefined as polynomial multiplication. The second definition wins from then on, so keep the names distinct (`multiply_polynomial` for scalar, `multiply_poly` for polynomial).
- `add_poly` is assumed to line up the lists from the right and pad the shorter one with zeros. Your version does this by reversing both lists.

---

## 8. Visual: recursive `solve_equations` step by step

### 8.1 The helper functions (your code)

```python
def multiply_polynomial(poly, num):
    return [c * num for c in poly]

def add_poly(p1, p2):
    p1 = p1[::-1]
    p2 = p2[::-1]
    ans = []
    if len(p1) > len(p2):
        for i in range(len(p1)):
            if i + 1 > len(p2):
                ans.append(p1[i])
            else:
                ans.append(p1[i] + p2[i])
    elif len(p1) == len(p2):
        for i in range(len(p1)):
            ans.append(p1[i] + p2[i])
    else:
        for i in range(len(p2)):
            if i + 1 > len(p1):
                ans.append(p2[i])
            else:
                ans.append(p1[i] + p2[i])
    return ans[::-1]

print(add_poly([2, 5, 6], [2, 5, 7]))
```

```python
def multiply_poly(p1, p2):
    result = [0]
    for i in range(len(p1)):
        c = p1[i]
        power = len(p1) - 1 - i
        scaled = multiply_polynomial(p2, c)
        shifted = scaled + [0] * power
        result = add_poly(result, shifted)
    return result

print(multiply_poly([2, -1, 3], [1, 2]))
```

```python
def solve_for_first_variable(arr, consts):
    rhs = arr[-1]
    for i in range(len(consts)):
        rhs -= arr[i + 1] * consts[i]
    return rhs / arr[0]

print(solve_for_first_variable([3, 4, 6, 20], [5, 6]))
```

```python
def eliminate_variable_pair(arr1, arr2, index):
    c1 = arr1[index] * (-1)
    c2 = arr2[index]
    p1 = multiply_polynomial(arr1, c2)
    p2 = multiply_polynomial(arr2, c1)
    diff_poly = add_poly(p1, p2)
    return diff_poly[:index] + diff_poly[index + 1:]

print(eliminate_variable_pair([2, 1, 5], [1, -1, 1], 0))
```

```python
def eliminate_variable(arr, index):
    pivot = arr[0]
    results = []
    for i in range(1, len(arr)):
        results.append(eliminate_variable_pair(pivot, arr[i], index))
    return results

print(eliminate_variable([[2, 1, 5], [1, -1, 1]], 0))
```

### 8.2 The recursive function

```python
def solve_equations(equations):
    if len(equations) == 1:
        return [equations[0][-1] / equations[0][0]]
    reduced = eliminate_variable(equations, 0)
    rest = solve_equations(reduced)
    x1 = solve_for_first_variable(equations[0], rest)
    return [x1] + rest
```

**The idea:** `solve_equations` solves a big system by shrinking it. Each call removes one variable, then asks itself to solve the smaller system. This is recursion.

**Going down (the first half):**

```python
if len(equations) == 1:
    return [equations[0][-1] / equations[0][0]]   # base case
reduced = eliminate_variable(equations, 0)         # remove the first variable
rest = solve_equations(reduced)                    # same function, smaller problem
```

- With 3 equations you get 2, then 1. The base case is the single equation `a*z = b`, which is solved by dividing `b / a`.
- The base case is what stops the recursion. Without it the function would call itself forever.

**Coming back up (the second half):**

```python
x1 = solve_for_first_variable(equations[0], rest)
return [x1] + rest
```

- When `rest` comes back, it holds the answers for all the *other* variables.
- Your original equation `equations[0]` still contains the first variable. Plug in the known values, solve for it, and put it at the front of the list.
- This is the same idea as back substitution. Here it happens as the recursive calls return.

**How the helpers fit**

- `eliminate_variable` takes the first equation as the pivot and combines it with every other equation. Each result has the first variable cancelled out, so those equations are one entry shorter.
- `eliminate_variable_pair` is where the cancelling happens. It multiplies each equation by the other's coefficient so the two coefficients match in size and are opposite in sign, then adds them. That's why the sum starts with a 0, which gets dropped.
- `solve_for_first_variable` computes `(rhs - other terms) / first coefficient`.

### 8.3 Trace

System:

```
 x +  y +  z = 6    ->  [1,  1,  1, 6]
2x -  y +  z = 3    ->  [2, -1,  1, 3]
 x + 2y -  z = 2    ->  [1,  2, -1, 2]
```

**Call 1** (equations = `[[1,1,1,6], [2,-1,1,3], [1,2,-1,2]]`, variables x, y, z). `len = 3`, not the base case, so eliminate x.

- `pivot = [1, 1, 1, 6]`
- Pair with `[2, -1, 1, 3]`: `c1 = -1`, `c2 = 2`
  - `pivot * 2 = [2, 2, 2, 12]`, `eq * -1 = [-2, 1, -1, -3]`
  - add = `[0, 3, 1, 9]`, drop index 0, result `[3, 1, 9]`
- Pair with `[1, 2, -1, 2]`: `c1 = -1`, `c2 = 1`
  - `pivot * 1 = [1, 1, 1, 6]`, `eq * -1 = [-1, -2, 1, -2]`
  - add = `[0, -1, 2, 4]`, drop index 0, result `[-1, 2, 4]`
- `reduced = [[3, 1, 9], [-1, 2, 4]]` (x is gone), then Call 2 starts.

**Call 2** (equations = `[[3,1,9], [-1,2,4]]`, variables y, z). `len = 2`, so eliminate y.

- `pivot = [3, 1, 9]`
- Pair with `[-1, 2, 4]`: `c1 = -3`, `c2 = -1`
  - `pivot * -1 = [-3, -1, -9]`, `eq * -3 = [3, -6, -12]`
  - add = `[0, -7, -21]`, drop index 0, result `[-7, -21]`
- `reduced = [[-7, -21]]` (y is gone), then Call 3 starts.

**Call 3** (equations = `[[-7, -21]]`, variable z). `len = 1`, the base case.

- The row means `-7z = -21`.
- `return [-21 / -7] = [3.0]`. So z = 3.

**Back in Call 2:** `rest = [3.0]`.

- `solve_for_first_variable([3, 1, 9], [3.0])`: `rhs = 9 - 1*3 = 6`, then `6 / 3 = 2.0`. So y = 2.
- `return [2.0] + [3.0] = [2.0, 3.0]`

**Back in Call 1:** `rest = [2.0, 3.0]`.

- `solve_for_first_variable([1, 1, 1, 6], [2.0, 3.0])`: `rhs = 6 - 1*2 - 1*3 = 1`, then `1 / 1 = 1.0`. So x = 1.
- `return [1.0] + [2.0, 3.0] = [1.0, 2.0, 3.0]`

**Answer: x = 1, y = 2, z = 3.**

Call stack summary:

| Step | Call | What happens | Returns |
|---|---|---|---|
| 1 | Call 1 | eliminate x, call Call 2 | (waiting) |
| 2 | Call 2 | eliminate y, call Call 3 | (waiting) |
| 3 | Call 3 | base case: -21 / -7 | `[3.0]` |
| 4 | Call 2 | solve y using z = 3 | `[2.0, 3.0]` |
| 5 | Call 1 | solve x using y, z | `[1.0, 2.0, 3.0]` |

**One thing to watch:** this code does no row swapping. If a pivot is 0 (for example, the first equation starts with `0`), it will divide by zero. Your test cases are fine, but a system like `[[0, 1, 2], [1, 1, 3]]` would break it.

---

## 9. Appendix: your notes

(Lightly cleaned for typos; content kept as you wrote it.)

### Python and basics

- Jupyter is a wrapper around Python, an interactive shell notebook.
- Log: counting powers of 10 in the decimal number system, a way of finding the power.
- `random.seed(1)` makes `random.random()` predictable.

### Machine learning basics

- The best fit is the line with minimum MSE for a non-linear set of points.
- If the problem statement is complex, we need more parameters.
- Models use MSE or MAE or MAE+MSE, like Huber loss (used for LLM training).
- Given x, we get y: inference, which is faster than training with historical data.
- Data is bigger than the model.
- From data, y, training gives us a model. Using the model we get inference. `m` and `c` are parameters.
- Dataset: for each data point we have one linear equation, with weights for multiple inputs and one bias. This is linear regression, a method of modeling systems, and supervised ML where we have multiple inputs and one output and try to figure out the relationship between them.
- Given a set of points x, I have a set of weights `w` and bias `b`, with which I can predict y.

### Data shapes

- Audio: 1D array. Grayscale image: 2D. Colored image: 3D. Video: 4D `(5000, height, width, 3)`, where 5000 is the number of color images (frames) and 3 values (RGB) per pixel.
- A grayscale image typically has 1 value per pixel, instead of 3 values like an RGB image.
- Grayscale video is 3D, color video is 4D.

**RGB vs Grayscale**

RGB: pixel = `[R, G, B]`, for example `[255, 0, 0]` is red. An RGB image has shape `(height, width, 3)`.

Grayscale: pixel = `[intensity]`. The single value represents how bright or dark the pixel is. Usually:

```
0   -> Black
255 -> White
128 -> Gray
```

```
0      50      128      200      255
|       |       |        |        |
Black  Dark    Gray    Light    White
```

A grayscale image has shape `(height, width)` rather than `(height, width, 3)`.

A tiny 2x2 grayscale image:

```python
gray = [
    [0,   128],
    [200, 255]
]
```

```
+---------+---------+
|    0    |   128   |
|  black  |  gray   |
+---------+---------+
|   200   |   255   |
|  light  |  white  |
+---------+---------+
```

- RGB: 3 values per pixel (R, G, B).
- Grayscale: 1 value per pixel (intensity).

One small detail: grayscale images can sometimes be stored with 3 identical channels (for example `[128,128,128]`), but conceptually and in the usual NumPy representation, a grayscale image is one value per pixel.

### Sum of 1 to n

```
S = 1 + 2 + ... + n
S = n + n-1 + ... + (n-(n-1))
S = n*n - (1 + 2 + ... + n-1)
S = n*n - (1 + 2 + ... + n-1 + n - n)
S = n*n - (S - n)
2S = n^2 + n
S = n*(n+1)/2
```

Unique handshakes in a room = `n*(n-1)/2`.

Other topic noted: logarithm using recursion.

### Combinations (ABCDEF example)

#### Step 1: What does 4x3x2 actually count?

The logic (position 1: 4 choices, position 2: 3 choices, position 3: 2 choices) gives 24, but it counts **ordered** triplets. ABC and BCA are counted as two *different* outcomes, even though you want them treated as the same group.

#### Step 2: Ask the same kind of question again, but smaller

Since the overcounting comes from order, ask: **for one fixed set of 3 letters, say {A, B, C}, how many different orderings can I make using only those 3 letters?**

- Position 1: 3 choices
- Position 2: 2 choices
- Position 3: 1 choice

So 3 x 2 x 1 = 6 orderings of the same 3 letters: **ABC, ACB, BAC, BCA, CAB, CBA**. All 6 represent the *same* triplet {A, B, C}.

#### Step 3: Every triplet gets repeated by exactly this same amount

This "6 orderings per group" holds for *any* group of 3 letters, because the argument never used which specific letters were in the group, only that there were 3 of them.

| Group | The 6 orderings |
|---|---|
| {A,B,C} | ABC, ACB, BAC, BCA, CAB, CBA |
| {A,B,D} | ABD, ADB, BAD, BDA, DAB, DBA |
| {A,C,D} | ACD, ADC, CAD, CDA, DAC, DCA |
| {B,C,D} | BCD, BDC, CBD, CDB, DBC, DCB |

4 groups x 6 orderings each = 24, which matches the permutation count.

#### Step 4: Divide out the repetition

$$\text{unique triplets} = \frac{\text{total orderings}}{\text{orderings per triplet}} = \frac{24}{6} = 4$$

This matches {A,B,C}, {A,B,D}, {A,C,D}, {B,C,D}, exactly what 4C3 gives.

#### Step 5: Generalize

- Total orderings (numerator) = n x (n-1) x ... x (n-r+1) = n!/(n-r)!
- Orderings per group (the repeat factor) = r x (r-1) x ... x 1 = r!

$$\binom{n}{r} = \frac{n!/(n-r)!}{r!} = \frac{n!}{r!(n-r)!}$$

The intuition: **permutations overcount by exactly r!, because that's how many ways you can shuffle any fixed set of r chosen items among themselves.** Dividing by it collapses every group of r! duplicate orderings down to the single unordered group they all represent.

### Systems of linear equations

**Types of systems (2 equations)**

| Equations | Information | Type | Solutions |
|---|---|---|---|
| 2 sentences, 2 info | complete, non-singular (most informative) | unique solution | one |
| 2 sentences, 1 info | redundant, singular | infinite solutions | many |
| 2 sentences, contradicting info | contradictory, singular | no solution | none |

- Linear equation: all variables are multiplied by a coefficient, plus a constant.

**3 variables: planes**

- If there is no constant `c`, the plane passes through the origin.
- 3 equations with 3 different pieces of info: non-singular, with a unique solution, a point where the 3 planes intersect.
- If 2 planes intersect in a line and the third also contains that line, there are infinitely many solutions: every point on that line.
- If all 3 equations are the same, all planes are the same plane, every point on it satisfies the system, so the solution is a plane.
- Another type: 3 planes where each pair intersects in a different line (3 lines in total) but there is no common line for all 3. The 3 lines are parallel to each other. No solution.

### Row echelon form, determinant, rank

- Row echelon form: an upper triangular matrix. Everything below the diagonal is zero. The diagonal may be zeros, ones, or a mix, and everything to the left of a leading entry in its row is zero.
- Row operations and the determinant:
  - Adding rows: same determinant.
  - Multiplying a row by a scalar: determinant times the scalar.
  - Switching rows: negative of the previous determinant.
  - These operations preserve singularity / non-singularity.
- Rank: the amount of information carried by a system of equations.
- The rank of a matrix is the number of ones on the diagonal in row echelon form.
- Identity matrix: ones along the diagonal.

### Polynomials

- Every function can be expressed as a polynomial.
- n data points need an (n-1) degree polynomial, which has n-2 bends.
