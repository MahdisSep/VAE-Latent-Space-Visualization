# 🌌 Variational Autoencoders (VAE) and Latent Space Analysis with PyTorch

## 📝 Overview
This project implements a **Variational Autoencoder (VAE)** using the PyTorch framework to learn a probabilistic representation of the input data. The primary goal is to explore the properties of the continuous **latent space** for generating new, meaningful samples and to visualize this space using **t-SNE**.

The model is trained on the **MNIST** dataset of handwritten digits, demonstrating the core concepts of generative modeling and dimensionality reduction.

## 🛠️ Technologies Used
* **Framework:** PyTorch
* **Language:** Python
* **Libraries:** `torchvision`, `NumPy`, `Matplotlib`, `scikit-learn` (`TSNE`), `scikit-image` (`PSNR`, `SSIM`)

## 🎯 Project Objectives

1.  **VAE Architecture Implementation:** Construct a VAE model in PyTorch, including the Encoder (to output $\mu$ and $\log\sigma^2$) and the Decoder.
2.  **Reparameterization Trick:** Implement the **Reparameterization Trick** to enable backpropagation through the sampling process from the Gaussian latent distribution.
3.  **Loss Function Definition:** Define the VAE loss, which combines the **Reconstruction Loss** (e.g., Binary Cross-Entropy or MSE) and the **KL Divergence Loss** (to regularize the latent space toward a standard normal distribution).
4.  **Latent Space Visualization:** Use **t-distributed Stochastic Neighbor Embedding (t-SNE)** to reduce the high-dimensional latent vectors (e.g., $D=64$) down to 2 dimensions for visualization, allowing for clustering analysis of the digit classes.
5.  **Reconstruction Assessment:** Quantitatively evaluate the VAE's ability to reconstruct the input images using metrics like **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index Measure).

## 🧠 VAE Architecture and Analysis

### VAE Structure
The model employs Convolutional layers in the Encoder and Transposed Convolutional layers in the Decoder. The latent space dimension is a key hyperparameter explored in the notebook (e.g., $D=2$ for direct visualization).

### Latent Space Visualization (t-SNE)
The output of the t-SNE plot is crucial, demonstrating:
* **Contiguity:** The latent space is smooth and continuous, confirming that interpolating between two points yields meaningful samples.
* **Clustering:** Points corresponding to the same digit class cluster together, validating that the VAE has learned discriminative features.


### Reconstruction Quality
The calculated **PSNR** and **SSIM** values provide an objective measure of the reconstructed image quality against the original input, affirming the effectiveness of the model's decoder.

## 📂 Repository Structure

```

├── VAE-Latent-Space-Visualization/
│   ├── README.md                 \<-- You are here
│   ├── HW3\_q1.ipynb              \<-- PyTorch VAE implementation, training, and visualization
│   └── hw3\_q1.py                 \<-- Python script containing the VAE class and main training/testing loops

````

### showcase

![images](https://github.com/MahdisSep/VAE-Latent-Space-Visualization/blob/main/results/results1.png)
![images](https://github.com/MahdisSep/VAE-Latent-Space-Visualization/blob/main/results/results2.png)
![images](https://github.com/MahdisSep/VAE-Latent-Space-Visualization/blob/main/results/results3.png)
![images](https://github.com/MahdisSep/VAE-Latent-Space-Visualization/blob/main/results/results4.png)
![images](https://github.com/MahdisSep/VAE-Latent-Space-Visualization/blob/main/results/results5.png)
![images](https://github.com/MahdisSep/VAE-Latent-Space-Visualization/blob/main/results/results6.png)

-----

## ⚙️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/MahdisSep/VAE-Latent-Space-Visualization.git]
    cd VAE-Latent-Space-Visualization
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy matplotlib scikit-learn scikit-image
    ```
3.  **Execute:** Run the `HW3_q1.ipynb` notebook in an environment supporting PyTorch (preferably with GPU acceleration) to load the MNIST data, train the VAE, generate new samples, and visualize the latent space using t-SNE.
````
