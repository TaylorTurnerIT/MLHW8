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
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Load MNIST
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_all = mnist.data.astype(float) / 255.0  # normalize to [0, 1]
    y_all = mnist.target.astype(int)

    # Binary label: 1 if digit is 6 or 7, else 0
    y_binary = ((y_all == 6) | (y_all == 7)).astype(int).reshape(-1, 1)

    # Split: 70 / 15 / 15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_binary, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Positive class rate: {y_binary.mean():.3f}")
    return X_train, fetch_openml, np, plt, train_test_split, y_train


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

    To calculate the point where the sigmoid activation is 0.5, we must first calculate the result of $\hat y$ and insert it into that function.

    Let $\hat y$ be
    $$\hat y = w_1 * input_1 + w_2 * input_2 + b$$

    Sigmoid activation function

    $$\frac{1}{1 + e^{-\hat y}}$$

    Plugging in the values for $\hat y$, we see

    $$\hat y = input_1 + input_2$$

    So the sigmoid becomes

    $$\frac{1}{1+e^{-(input_1 + input_2)}}$$

    To calculate the value at 0.5, we make the equation equal to that value and solve for the inputs.

    $$0.5 = \frac{1}{1+e^{-(input_1 + input_2)}}$$
    $$\implies 1 = 0.5 * 1+e^{-(input_1 + input_2)}$$
    $$\implies 2 = 1 + e^{-(input_1 + input_2)}$$
    $$\implies 1 = e^{-(input_1 + input_2)}$$
    $$\implies ln(1) = -(input_1 + input_2)$$
    $$\implies 0 = -(input_1 + input_2)$$
    $$\implies 0 = input_1 + input_2$$

    With this in mind, we can generalize and say that when $\hat y = 0$, the result of the sigmoid activation function is zero.

    2. Plot this boundary on a set of axes with $input_1$ on the x-axis and $input_2$ on the y-axis, for values ranging from $-2$ to $2$. Label the region where the neuron outputs greater than 0.5 and the region where it outputs less than 0.5.
    """)
    return


@app.cell
def _(np, plt):
    # 1. Define the range for input_1 from -2 to 2
    input_1 = np.linspace(-2, 2, 100)

    # 2. Calculate input_2 based on the decision boundary equation: input_2 = -input_1
    input_2 = -input_1

    # 3. Create the plot
    plt.figure(figsize=(8, 8))

    # Plot the decision boundary line
    plt.plot(
        input_1,
        input_2,
        color="black",
        linewidth=2,
        label="Decision Boundary ($\hat{y} = 0$)",
    )

    # 4. Fill the regions
    # The region where output > 0.5 is where input_1 + input_2 > 0 (or input_2 > -input_1)
    plt.fill_between(
        input_1, input_2, 2, color="blue", alpha=0.2, label="Output > 0.5"
    )

    # The region where output < 0.5 is where input_1 + input_2 < 0 (or input_2 < -input_1)
    plt.fill_between(
        input_1, -2, input_2, color="red", alpha=0.2, label="Output < 0.5"
    )

    # 5. Format the axes and labels
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("$input_1$")
    plt.ylabel("$input_2$")
    plt.title("Single Neuron Decision Boundary")

    # Add origin axes for better readability
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add the legend
    plt.legend(loc="upper right")

    # Show the plot
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    3. In 2–3 sentences, explain what this boundary would look like if the neuron had 3 inputs instead of 2. How does the geometry change?

    We would see a new point of control to modify the line, but no change to the shape. If we saw something like $x^2$ instead that would start to modify geometry.

    ### Part 1b — Linear Separability (10 pts)
    There are four datasets below shown in the figure. For each one, state whether a single neuron could learn to perfectly separate the two classes, and explain your reasoning in 1–2 sentences.

    no, yes, yes, no

    1. For the first we cannot draw any line that separates a 2D projection of a doughnut.
    2. Yes, we can draw a 2D line that could act as a plane by making the change in $z$ nothing.
    3. Yes, proof by above plot.
    4. No, there is a non-linear gap between the two galaxy clusters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="image1.png",
        alt="Which of these can a single neuron solve perfectly?",
        width=500,  # Optional: adjust width as needed
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part 1c — Symmetry and Identical Initialization (5 pts)
    Consider a neural network with 2 neurons in the first layer and 1 neuron in the second layer. All weights and biases in the entire network are initialized to 0. After 1 epoch of training, the second neuron in the first layer has learned weights `[1, 2]` as shown in the image below.

    What are the weights of the first neuron in the first layer after that same epoch? State your answer and explain why in 3–4 sentences.

    Since the initial weights were set to 0 and there was no random modification, they would deterministicly settle on the same local minima of [1,2]. Thus, we would have identical weights for both nodules.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="image2.png", alt="Layer 1 Initialization Diagram")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="image3.png",
        alt="Part 2 Network — All weights, biases, and inputs provided",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — Backpropagation by Hand (35 pts)
    This part must be completed entirely by hand. You may write your work on paper and photograph it, or typeset it — either is acceptable. **No code for this part.**

    ### The Network
    You are given a two-layer network with the following parameters:

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


    class Neuron:
        def __init__(self, n_features: int):
            """
            Initialize weights and bias.
            W : np.ndarray, shape (1, n_features)  small random values
            b : float, initialized to 0
            """
            self.W = np.random.randn(1, n_features) * 0.01
            self.b: float = 0.0

        def forward(self, X: np.ndarray) -> np.ndarray:
            self.X = X
            # Assert input shape
            assert X.shape[1] == self.W.shape[1], (
                f"X has {X.shape[1]} features, but W expects {self.W.shape[1]}"
            )

            Y_hat = X @ self.W.T + self.b
            assert Y_hat.shape == (X.shape[0], 1), (
                f"Y_hat shape {Y_hat.shape} != expected ({X.shape[0]}, 1)"
            )

            self.P_hat = 1 / (1 + np.exp(-Y_hat))  # overflow
            self.P_hat = np.clip(self.P_hat, 1e-9, 1 - 1e-9)

            # Assert output shape
            assert self.P_hat.shape == (X.shape[0], 1), (
                f"P_hat shape {self.P_hat.shape} is incorrect"
            )
            return self.P_hat

        def backward(self, dL_dP_hat: np.ndarray, lr: float):
            # Assert incoming gradient shape
            assert dL_dP_hat.shape == self.P_hat.shape, (
                f"dL_dP_hat shape {dL_dP_hat.shape} doesn't match P_hat shape {self.P_hat.shape}"
            )

            # Compute sigmoid derivative term
            sigmoid_grad = dL_dP_hat * self.P_hat * (1 - self.P_hat)
            assert sigmoid_grad.shape == self.P_hat.shape

            # Gradient w.r.t. weights
            dL_dW = (
                sigmoid_grad.T @ self.X / self.X.shape[0]
            )  # shape (1, n_features)
            assert dL_dW.shape == self.W.shape, (
                f"dL_dW shape {dL_dW.shape} != W shape {self.W.shape}"
            )

            # Gradient w.r.t. bias
            dL_db = np.mean(sigmoid_grad)  # scalar

            # Update weights and bias
            self.W = self.W - lr * dL_dW
            self.b = self.b - lr * dL_db


    class Layer:
        def __init__(self, n_neurons, n_features, random_seed=42):
            self.neurons = []
            for neuron in range(n_neurons):
                self.neurons.append(Neuron(n_features))

        def forward(self, X):
            # Sigmoid activated inferences
            self.inferences = []
            for neuron in self.neurons:
                self.inferences.append(neuron.forward(X))
            return np.hstack(self.inferences)

        def backward(self, dL_dP_hat, lr):
            for neuron in self.neurons:
                neuron.backward(dL_dP_hat, lr)

    return (Layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part 3b — Batch Size Experiment on MNIST (20 pts)
    Using the network architecture and hyperparameters below, train four separate models on MNIST — one for each batch size. The only difference between the four models should be the batch size.

    * For "full" batch, pass the entire training set in a single forward/backward pass each epoch (standard gradient descent).
    * For all others, shuffle the training data at the start of each epoch and process it in mini-batches of the given size, performing one weight update per batch (stochastic gradient descent).

    For each model, record training loss and validation loss each epoch. Then answer the following:
    1. Plot all four training loss curves on one set of axes and all four validation loss curves on another. Use a legend to identify each batch size.
    2. How does convergence speed change as batch size decreases? Why does this change occur?
    3. Do any of the models show signs of overfitting?
    """)
    return


@app.cell
def _(Layer, X_train, np, y_train):
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import seaborn as sns

    np.random.seed(42)
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Load MNIST
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_all = mnist.data.astype(float) / 255.0  # normalize to [0, 1]
    y_all = mnist.target.astype(int)

    # Binary label: 1 if digit is 6 or 7, else 0
    y_binary = ((y_all == 6) | (y_all == 7)).astype(int).reshape(-1, 1)

    # Split: 70 / 15 / 15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_binary, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Positive class rate: {y_binary.mean():.3f}")
    """

    # Architecture: 1 hidden layer
    layer1_neurons = 10
    layer2_neurons = 1

    lr = 0.1
    n_epochs = 120
    batch_sizes = ["full", 128, 32, 8]  # train once for each
    n_features = X_train.shape[1]
    # Train: (49000, 784), Val: (10500, 784), Test: (10500, 784)


    def get_batches(X, y, batch_size):
        """Yield (X_batch, y_batch) slices. 'full' yields the whole dataset once."""
        n = X.shape[0]
        if batch_size == "full":
            yield X, y
            return
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield X[idx], y[idx]


    for batch_size in batch_sizes:
        layers = [
            Layer(n_neurons=layer1_neurons, n_features=n_features),
            Layer(n_neurons=layer2_neurons, n_features=layer1_neurons),
        ]

        for epoch in range(n_epochs):
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                # --- forward pass ---
                out = X_batch
                for layer in layers:
                    out = layer.forward(out)

                # --- backward pass ---
                dL_dP = (out - y_batch) / y_batch.shape[0]
                for layer in reversed(layers):
                    layer.backward(dL_dP, lr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(Layer, fetch_openml, np, train_test_split):
    def _():
        from sklearn.preprocessing import MinMaxScaler

        # --- Load & prep ---
        magic = fetch_openml("magic", version=1, as_frame=False, parser="auto")
        X_magic = magic.data.astype(float)
        y_magic = (
            (magic.target == "g").astype(int).reshape(-1, 1)
        )  # gamma=1, hadron=0

        X_temp_m, X_test_m, y_temp_m, y_test_m = train_test_split(
            X_magic, y_magic, test_size=0.15, random_state=42
        )
        X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
            X_temp_m, y_temp_m, test_size=0.15 / 0.85, random_state=42
        )

        scaler = MinMaxScaler()
        X_train_m = scaler.fit_transform(X_train_m)  # fit on train only
        X_val_m = scaler.transform(X_val_m)
        X_test_m = scaler.transform(X_test_m)

        # --- Hyperparameters ---
        layer1_neurons = 60
        layer2_neurons = 25
        layer3_neurons = 1
        lr_magic = 0.01
        n_epochs_m = 600
        batch_size_m = 64
        n_features_m = X_train_m.shape[1]  # 10

        # --- Helpers ---
        def forward_pass(layers, X):
            out = X
            for layer in layers:
                out = layer.forward(out)
            return out

        def compute_loss(p_hat, y):
            return -np.mean(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

        def get_batches(X, y, batch_size):
            n = X.shape[0]
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                yield X[idx], y[idx]

        # --- Build layers ---
        np.random.seed(42)
        layers_magic = [
            Layer(n_neurons=layer1_neurons, n_features=n_features_m),
            Layer(n_neurons=layer2_neurons, n_features=layer1_neurons),
            Layer(n_neurons=layer3_neurons, n_features=layer2_neurons),
        ]

        train_losses, val_losses = [], []

        for epoch in range(n_epochs_m):
            for X_batch, y_batch in get_batches(
                X_train_m, y_train_m, batch_size_m
            ):
                out = forward_pass(layers_magic, X_batch)
                dL_dP = (out - y_batch) / y_batch.shape[0]
                for layer in reversed(layers_magic):
                    layer.backward(dL_dP, lr_magic)

            train_p = forward_pass(layers_magic, X_train_m)
            val_p = forward_pass(layers_magic, X_val_m)
            train_losses.append(compute_loss(train_p, y_train_m))
            val_losses.append(compute_loss(val_p, y_val_m))

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs_m} — train loss: {train_losses[-1]:.4f}  val loss: {val_losses[-1]:.4f}"
                )

        return train_losses, val_losses


    _()
    return


@app.cell
def _(
    X_test_m,
    accuracy_score,
    auc,
    f1_score,
    forward_pass,
    layers_magic,
    plt,
    precision_score,
    recall_score,
    roc_curve,
    train_losses,
    val_losses,
    y_test_m,
):
    # --- Task 2: Loss curves ---
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("MAGIC — Training & Validation Loss (batch_size=64)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # The curves are jagged because each mini-batch of 64 samples gives a slightly
    # different gradient estimate — some batches are harder or easier than average,
    # causing the loss to oscillate around the true gradient direction every epoch.
    # Smaller batches produce noisier estimates, so the update direction fluctuates
    # more than it would with the full dataset.

    # --- Task 3: Metrics ---
    test_p = forward_pass(layers_magic, X_test_m)
    test_pred = (test_p >= 0.5).astype(int)

    acc = accuracy_score(y_test_m, test_pred)
    prec = precision_score(y_test_m, test_pred)
    rec = recall_score(y_test_m, test_pred)
    f1 = f1_score(y_test_m, test_pred)

    # Fill in your Assignment 7 single-neuron results below
    a7_acc, a7_prec, a7_rec, a7_f1 = 0.000, 0.000, 0.000, 0.000  # <-- replace

    print("\n── Test Set Comparison ──────────────────────────────────")
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(
        f"{'A7 Single Neuron':<25} {a7_acc:>6.3f} {a7_prec:>6.3f} {a7_rec:>6.3f} {a7_f1:>6.3f}"
    )
    print(
        f"{'3-Layer NN (60-25-1)':<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}"
    )

    # --- Task 4: ROC / AUC ---
    fpr, tpr, _ = roc_curve(y_test_m, test_p)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"3-Layer NN (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MAGIC — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
