# Neural Network Manual Exercise (Backpropagation)

This exercise verifies your ability to perform one full iteration of **Stochastic Gradient Descent (SGD)** on a simple network.

## Problem Statement
**Architecture:** 2 Input Neurons $\to$ 1 Hidden Neuron $\to$ 1 Output Neuron.
**Activation Function:** Sigmoid, $g(z) = \frac{1}{1+e^{-z}}$.
**Loss Function:** Squared Error, $E = \frac{1}{2}(y - o)^2$.
**Learning Rate:** $\alpha = 0.5$.

**Initial Weights:**
*   Input to Hidden ($W^{(1)}$): $w_{11} = 0.5$, $w_{21} = 0.5$, bias $b_1 = 0$.
*   Hidden to Output ($W^{(2)}$): $w_{h} = 1.0$, bias $b_2 = -1.0$.

**Training Example:**
*   Input: $x = [1, 0]$
*   Target: $y = 1$

## Tasks
1.  **Forward Pass:** Calculate the output $o$.
2.  **Error Calculation:** Calculate loss $E$.
3.  **Backward Pass:** Calculate gradients $\frac{\partial E}{\partial w_h}$ and $\frac{\partial E}{\partial w_{11}}$.
4.  **Update:** Calculate new weights $w_h^{new}$.

---

## Solution Space (Try it first!)

...
...
...

## Solution Key

### 1. Forward Pass
*   **Hidden Neuron ($h$):**
    *   $z_h = w_{11}x_1 + w_{21}x_2 + b_1 = (0.5)(1) + (0.5)(0) + 0 = 0.5$
    *   $a_h = \text{sigmoid}(0.5) \approx \mathbf{0.622}$
*   **Output Neuron ($o$):**
    *   $z_o = w_h a_h + b_2 = (1.0)(0.622) - 1.0 = -0.378$
    *   $o = \text{sigmoid}(-0.378) \approx \mathbf{0.406}$

### 2. Error
*   $E = \frac{1}{2}(1 - 0.406)^2 = \frac{1}{2}(0.594)^2 \approx \mathbf{0.176}$

### 3. Backward Pass (Gradients)
Recall Gradient Formula: $\delta \cdot \text{input}$.
*   **Output Layer Error ($\delta_o$):**
    *   $\delta_o = -(y - o) \cdot o(1-o)$  *(Derivative of loss * Derivative of sigmoid)*
    *   $\delta_o = -(1 - 0.406) \cdot (0.406)(1 - 0.406)$
    *   $\delta_o = -0.594 \cdot 0.241 \approx \mathbf{-0.143}$
    *   **Gradient for $w_h$:** $\delta_o \cdot a_h = (-0.143)(0.622) \approx \mathbf{-0.089}$

*   **Hidden Layer Error ($\delta_h$):**
    *   $\delta_h = (\delta_o \cdot w_h) \cdot a_h(1-a_h)$
    *   $\delta_h = (-0.143 \cdot 1.0) \cdot (0.622)(1-0.622)$
    *   $\delta_h = -0.143 \cdot 0.235 \approx \mathbf{-0.0336}$
    *   **Gradient for $w_{11}$:** $\delta_h \cdot x_1 = (-0.0336)(1) = \mathbf{-0.0336}$

### 4. Weight Update (Example for $w_h$)
*   $w_h^{new} = w_h - \alpha \cdot \frac{\partial E}{\partial w_h}$
*   $w_h^{new} = 1.0 - 0.5(-0.089)$
*   $w_h^{new} = 1.0 + 0.0445 = \mathbf{1.0445}$
