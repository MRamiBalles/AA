# K-Means Manual Exercise (Exam Style)

This exercise tests your ability to execute the **K-Means** algorithm step-by-step. This is a very common exam question.

## Problem Statement
Given the following 4 data points in 2D space:
*   $A = (1, 1)$
*   $B = (2, 1)$
*   $C = (4, 3)$
*   $D = (5, 4)$

And **$K=2$** clusters.
Initialize the centroids to be points **A** and **C**:
*   $\mu_1 = (1, 1)$
*   $\mu_2 = (4, 3)$

## Tasks
1.  **Assignment Step:** Calculate the Euclidean distance from each point to both centroids. Assign each point to the closest cluster.
2.  **Update Step:** Calculate the new coordinates of the centroids ($\mu_1', \mu_2'$) based on the mean of the points assigned to them.

---

## Solution Space (Try it first!)

...
...
...

## Solution Key

### 1. Assignment Step
Calculate distances ($d(P, \mu) = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$):

*   **Point A (1,1):**
    *   $d(A, \mu_1) = 0$
    *   $d(A, \mu_2) = \sqrt{(1-4)^2 + (1-3)^2} = \sqrt{9+4} \approx 3.6$
    *   $\to$ Assign to **Cluster 1**.

*   **Point B (2,1):**
    *   $d(B, \mu_1) = \sqrt{(2-1)^2 + (1-1)^2} = 1$
    *   $d(B, \mu_2) = \sqrt{(2-4)^2 + (1-3)^2} = \sqrt{4+4} \approx 2.8$
    *   $\to$ Assign to **Cluster 1** ($1 < 2.8$).

*   **Point C (4,3):**
    *   $d(C, \mu_1) \approx 3.6$
    *   $d(C, \mu_2) = 0$
    *   $\to$ Assign to **Cluster 2**.

*   **Point D (5,4):**
    *   $d(D, \mu_1) = \sqrt{(5-1)^2 + (4-1)^2} = \sqrt{16+9} = 5$
    *   $d(D, \mu_2) = \sqrt{(5-4)^2 + (4-3)^2} = \sqrt{1+1} \approx 1.4$
    *   $\to$ Assign to **Cluster 2**.

**Assignments:** $C_1 = \{A, B\}$, $C_2 = \{C, D\}$.

### 2. Update Step (New Centroids)
Calculate the mean of the points in each cluster:

*   **New $\mu_1$:** Mean of $A(1,1)$ and $B(2,1)$.
    *   $x = \frac{1+2}{2} = 1.5$
    *   $y = \frac{1+1}{2} = 1$
    *   $\mathbf{\mu_1' = (1.5, 1)}$

*   **New $\mu_2$:** Mean of $C(4,3)$ and $D(5,4)$.
    *   $x = \frac{4+5}{2} = 4.5$
    *   $y = \frac{3+4}{2} = 3.5$
    *   $\mathbf{\mu_2' = (4.5, 3.5)}$
