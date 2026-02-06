# SVM Manual Exercise (Exam Style)

Since we cannot extract the exact text from the PDF, this exercise replicates a standard "Support Vector Machine" exam problem.

## Problem Statement
Consider a simple 2-dimensional dataset with 3 points:

*   **Positive Class ($y=+1$):**
    *   $x^{(1)} = (2, 0)$
    *   $x^{(2)} = (0, 2)$
*   **Negative Class ($y=-1$):**
    *   $x^{(3)} = (0, 0)$

## Questions

1.  **Visualization:** Sketch the points on a 2D plane. Can you see the line that separates them with the maximum margin?
2.  **Support Vectors:** Which points are the "Support Vectors"? (The hardest ones to classify, closest to the boundary).
3.  **Hyperplane:** Find the Maximum Margin Hyperplane defined by $w \cdot x + b = 0$.
    *   *Hint:* By symmetry, the normal vector $w$ should point towards $(1,1)$.
4.  **Margin:** Calculate the geometric margin $\gamma = \frac{2}{||w||}$.

## Solution Steps (Try first!)

...
...
...

### Solution Key for Verification

1.  **Support Vectors:** All three points are support vectors. $(0,0)$ pushes from below, and $(2,0), (0,2)$ push from above.
2.  **Hyperplane:**
    *   The line passes exactly between $(0,0)$ and the segment connecting $(2,0)-(0,2)$.
    *   Midpoint between classes is $(0.5, 0.5)$.
    *   Equation: $x_1 + x_2 - 1 = 0$.
    *   Therefore: $w = (1, 1)$, $b = -1$.
3.  **Verification:**
    *   $w \cdot x^{(3)} + b = (1)(0) + (1)(0) - 1 = -1$ (Matches $y=-1$)
    *   $w \cdot x^{(1)} + b = (1)(2) + (1)(0) - 1 = 1$ (Matches $y=+1$)
4.  **Margin:**
    *   $||w|| = \sqrt{1^2 + 1^2} = \sqrt{2}$
    *   Margin $= \frac{2}{\sqrt{2}} = \sqrt{2} \approx 1.41$
