import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning — Programming Assignment 8
    ## Multi-Layer Neural Networks: Automatic Feature Learning

    | Requirement | Details |
    | :--- | :--- |
    | **Points (4xxx)** | 115 pts (+30 bonus) |
    | **Points (5xxx)** | 145 pts (Going Beyond required) |
    | **Language** | Python 3.8+ |
    | **Datasets** | MNIST, MAGIC Gamma Telescope |
    | **Submission** | Single PDF via LMS |
    | **Seed / Split** | 42 / 70-15-15 train-val-test |

    ---

    ## Context and Motivation

    ### Where We Have Been
    In Assignment 7 you built a single artificial neuron from scratch and trained it on the MAGIC Gamma Telescope dataset, achieving around 80% accuracy.

    That neuron draws a hyperplane through the feature space. No matter how its weights are tuned, it can only ever produce a linear decision boundary. This is not a limitation of how it is trained — it is a fundamental geometric constraint.

    In this assignment you will stack neurons into layers, where the output of one layer becomes the input to the next. Each neuron in a hidden layer learns to detect a different pattern in the input, and the output layer combines those signals to make a final prediction. No manual feature engineering is required — the network discovers useful representations on its own through backpropagation.

    ### What You Will Build
    You will first work through the mathematics of a two-layer network entirely by hand — forward pass, loss, backpropagation, and weight updates — on a small network with provided weights.

    You will then implement a reusable `Layer` class and use it to train networks of different sizes on the MNIST handwritten digit dataset, exploring how batch size affects convergence and overfitting.

    Finally, you will apply your network to the MAGIC dataset from Assignment 7 and compare performance to your single-neuron baseline.

    ---

    ## Learning Objectives
    By the end of this assignment, you will be able to:
    * Explain geometrically why a single neuron cannot solve non-linearly separable problems.
    * Perform a full forward pass, compute cross-entropy loss, backpropagate gradients through two layers, and update all weights by hand.
    * *(Going Beyond — 5xxx required)* Verify a hand-computed gradient using the epsilon perturbation method and explain what this confirms.
    * Implement a reusable `Layer` class that handles forward inference and backward gradient computation.
    * Implement mini-batch SGD and explain why it provides more gradient feedback per epoch than batch gradient descent.
    * Train networks with different batch sizes and describe the effect on convergence speed and overfitting.
    * Evaluate a model using accuracy, precision, recall, F1, ROC curve, and AUC.
    * *(Going Beyond — 5xxx required)* Implement SGD with momentum and explain why it produces a smoother learning curve than standard SGD.

    ---

    ## LLM Usage
    * LLMs should **NOT** be used to solve the problems or implement the methods.
    * LLMs **CAN** be used to help generate clean plots or to understand syntax.

    ---

    ## Hyperparameters
    Default hyperparameters unless a part explicitly states otherwise:

    | Parameter | Value |
    | :--- | :--- |
    | `random_seed` | 42 |
    | `train/val/test` | 70 / 15 / 15 |

    Part-specific hyperparameters are stated at the start of each part.

    ---

    ## Datasets

    ### MNIST 6 vs. 7
    Use exactly the following code to load the MNIST dataset. Do not modify it. The data is already normalized to [0, 1]. Do not apply any additional scaling.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import seaborn as sns

    np.random.seed(42)
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)

    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_all = mnist.data.astype(float) / 255.0   # normalize to [0, 1]
    y_all = mnist.target.astype(int)

    # Binary label: 1 if digit is 6 or 7, else 0
    y_binary = ((y_all == 6) | (y_all == 7)).astype(int).reshape(-1, 1)

    # Split: 70 / 15 / 15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_binary, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42)

    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    print(f'Positive class rate: {y_binary.mean():.3f}')
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MAGIC Gamma Telescope
    Use the same loading code from Assignment 7. Apply the same 70/15/15 split and MinMaxScaler protocol, fitting the scaler on training data only.

    ---

    ## Part 1 — What Does a Single Neuron Do? (25 pts)

    ### Part 1a — Decision Boundary (10 pts)
    Consider a single neuron with 2 inputs and the following parameters:

    | Parameter | Value |
    | :--- | :--- |
    | `weights` | `[1, 1]` |
    | `bias` | `0` |

    Answer the following. Show all work.
    1. The decision boundary of a neuron is the set of inputs where its output (the sigmoid activation) is exactly 0.5. Give the equation of the decision boundary for this neuron. Your answer should be an equation relating $input_1$ and $input_2$.
    2. Plot this boundary on a set of axes with $input_1$ on the x-axis and $input_2$ on the y-axis, for values ranging from $-2$ to $2$. Label the region where the neuron outputs greater than 0.5 and the region where it outputs less than 0.5.
    3. In 2–3 sentences, explain what this boundary would look like if the neuron had 3 inputs instead of 2. How does the geometry change?

    ### Part 1b — Linear Separability (10 pts)
    There are four datasets below shown in the figure. For each one, state whether a single neuron could learn to perfectly separate the two classes, and explain your reasoning in 1–2 sentences.

    ![Which of these can a single neuron solve perfectly?](word/media/image1.png)

    ### Part 1c — Symmetry and Identical Initialization (5 pts)
    Consider a neural network with 2 neurons in the first layer and 1 neuron in the second layer. All weights and biases in the entire network are initialized to 0. After 1 epoch of training, the second neuron in the first layer has learned weights `[1, 2]` as shown in the image below.

    What are the weights of the first neuron in the first layer after that same epoch? State your answer and explain why in 3–4 sentences.

    ![Layer 1 Initialization Diagram](word/media/image2.png)

    ---

    ## Part 2 — Backpropagation by Hand (35 pts)
    This part must be completed entirely by hand. You may write your work on paper and photograph it, or typeset it — either is acceptable. **No code for this part.**

    ### The Network
    You are given a two-layer network with the following parameters:

    ![Part 2 Network — All weights, biases, and inputs provided](word/media/image3.png)

    | Component | Values |
    | :--- | :--- |
    | **Inputs and true label** | $x_1 = 0.5$, $x_2 = 0.8$, $y = 1$ |
    | **Layer 1: Neuron 1** | $w^{(1)}_{11} = 0.3$, $w^{(1)}_{12} = 0.5$, $b^{(1)}_1 = 0.1$ |
    | **Layer 1: Neuron 2** | $w^{(1)}_{21} = -0.2$, $w^{(1)}_{22} = 0.4$, $b^{(1)}_2 = -0.1$ |
    | **Layer 2: 1 neuron, 2 inputs** | $w^{(2)}_1 = 0.6$, $w^{(2)}_2 = -0.3$, $b^{(2)} = 0.2$ |
    | **Learning Rate** | $0.1$ |

    * The notation $w^{(1)}_{ij}$ means the weight in Layer 1 connecting input $j$ to neuron $i$.
    * The activation function is sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

    ### Your Tasks

    #### Part 2a — Forward Pass (10 pts)
    Compute the following values, showing every step of your arithmetic:
    * $y_{11}$ and $p_{11}$: the pre-activation and activation of Layer 1, Neuron 1
    * $y_{12}$ and $p_{12}$: the pre-activation and activation of Layer 1, Neuron 2
    * $y_2$ and $p_2$: the pre-activation and activation of Layer 2
    * $L$: the binary cross-entropy loss

    $$L = -[y \cdot \log(\hat{p}) + (1-y) \cdot \log(1-\hat{p})]$$

    #### Part 2b — Backward Pass and Weight Updates (20 pts)
    Starting from $\frac{\partial L}{\partial \hat{p}_2}$, backpropagate through the entire network and compute the gradient of the loss with respect to every weight and bias. Then compute all updated weights using `learning_rate = 0.1`. You must show all intermediate gradients.

    #### Going Beyond — Epsilon verification (10 points)
    *Required for 5xxx students | +20 bonus points for 4xxx students*

    Verify your hand-computed gradient for $w^{(1)}_{11}$ using the epsilon perturbation method.
    1. Use the weights described in part 2 without the gradient update above. Compute the loss $L$ on the single data point.
    2. Recompute the entire forward pass with $w^{(1)}_{11}$ replaced by $w^{(1)}_{11} + 0.001$, keeping all other weights at their original values.
    3. Compute the new loss $L_{new}$ with just this small change to the single weight.
    4. Compute the change in Loss divided by the change in weight: $\frac{L_{new} - L}{0.001}$.
    5. Compare this to the analytical gradient $\frac{\partial L}{\partial w^{(1)}_{11}}$ that you found in Part 2b. They should agree to at least 3 decimal places.
    6. In 2–3 sentences, explain how this value is related to the derivative $\frac{\partial L}{\partial w^{(1)}_{11}}$ which is the change in loss w.r.t $w^{(1)}_{11}$.
    7. If your loss computation in 2B had been incorrect, would it have been very similar to the value given by $\frac{L_{new} - L}{0.001}$? Explain why or why not.

    ---

    ## Part 3 — Implementation and Training (35 pts)

    ### Part 3a — Implement the Layer Class (15 pts)
    Complete the following class skeleton. Do not change any method signature.
    """)
    return


@app.cell
def _(np):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    class Layer:
        def __init__(self, n_neurons, n_features, random_seed=42):
            pass

        def forward(self, X):
            pass

        def backward(self, dL_dP_hat, lr):
            pass


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part 3b — Batch Size Experiment on MNIST (20 pts)
    Using the network architecture and hyperparameters below, train four separate models on MNIST — one for each batch size. The only difference between the four models should be the batch size.

    ```python
    # Architecture: 1 hidden layer
    layer1_neurons = 10
    layer2_neurons = 1

    lr = 0.1
    n_epochs = 120
    batch_sizes = ['full', 128, 32, 8]   # train once for each
    ```

    * For "full" batch, pass the entire training set in a single forward/backward pass each epoch (standard gradient descent).
    * For all others, shuffle the training data at the start of each epoch and process it in mini-batches of the given size, performing one weight update per batch (stochastic gradient descent).

    For each model, record training loss and validation loss each epoch. Then answer the following:
    1. Plot all four training loss curves on one set of axes and all four validation loss curves on another. Use a legend to identify each batch size.
    2. How does convergence speed change as batch size decreases? Why does this change occur?
    3. Do any of the models show signs of overfitting?

    ---

    ## Part 4 — MAGIC Gamma Telescope (25 pts)
    Apply your neural network to the MAGIC Gamma Telescope dataset from Assignment 7. You already have a single-neuron logistic regression baseline from that assignment — this part asks whether a multi-layer network does better.

    ### Hyperparameters and Architecture
    ```python
    layer1_neurons = 60
    layer2_neurons = 25
    layer3_neurons = 1

    lr = 0.01
    n_epochs = 600
    batch_size = 64
    ```

    ### Your Tasks
    1. Train the network on MAGIC using SGD with the hyperparameters above.
    2. Plot the training and validation loss curves. You will notice these are jagged. In 2–3 sentences, describe why the loss function is jagged.
    3. Report accuracy, precision, recall, and F1 on the test set. Present these alongside your Assignment 7 single-neuron results in a comparison table.
    4. Plot the ROC curve for your best model and report the AUC.
    5. Did the neural network improve over logistic regression? How large is the gain?
    6. In Assignment 7 you added degree-2 polynomial features to try to improve over the single-neuron baseline. Compare the test performance of that model to your neural network here. Which performed better?
    7. In 3–4 sentences, reflect on why polynomial features improved the single neuron and why the multi-layer neural network improved over both.

    ---

    ## Going Beyond — SGD with Momentum
    *Required for 5xxx students | +20 bonus points for 4xxx students*

    ### Background
    With mini-batch SGD, the loss curve on MAGIC is visibly jagged because each batch of 64 examples gives a slightly different gradient estimate. Momentum smooths this out by maintaining a running weighted average of past gradients, called the velocity.

    The weight update rule becomes:
    ```python
    v_W = beta * v_W + (1 - beta) * dL_dW   # velocity update
    W   = W - lr * v_W                      # weight update

    # Same for biases
    v_b = beta * v_b + (1 - beta) * dL_db
    b   = b - lr * v_b

    beta = 0.9   # momentum coefficient
    # v_W and v_b initialized to zeros, same shape as W and b
    ```

    ### Your Tasks
    1. Adjust the layer class as needed to create a `Layer_momentum` class that uses momentum based weight updates.
    2. Train the 60-25-1 network on MAGIC with SGD and momentum using hyperparameters (`lr=0.01`, `batch_size=64`, `epochs=600`, `beta=0.9`).
    3. Compare to the results in part 4. Report accuracy, precision, recall, and F1 for both. Plot the ROC curve and report AUC for the better of the two.
    4. In 3–4 sentences, describe the difference in the learning curves. Is the momentum curve smoother? Does momentum improve final performance?

    ---

    ## Submission Instructions
    1. Submit a single PDF.
    2. Your PDF must contain all written answers, all code, and all plots in one document. Use a Jupyter Notebook exported to PDF, or a Python script with outputs captured — whichever you prefer.
    3. All written answers must appear as clearly labeled prose directly below the relevant code or plot, not in a separate section at the end.
    4. Part 2 hand calculations may be photographed and included in the PDF.
    5. **5xxx students:** Going Beyond implementation and written answers must appear as clearly labeled prose immediately above the corresponding code.

    *Solutions will be released once the assignment closes. Grading reflects good-faith effort — a sincere attempt that demonstrates genuine engagement with the material will receive full credit even if the implementation is not perfectly correct.*
    """)
    return


if __name__ == "__main__":
    app.run()
