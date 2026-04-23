import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from opensimplex import OpenSimplex

def SimplexNoise(
    batch_size,
    height,
    width,
    t_slice=100,
    octaves=6,
    base_frequency=2**-5,
    decay=0.8,
    device="cpu",
):
    noise_batch = []
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    for b in range(batch_size):
        seed = np.random.randint(0, 1_000_000)
        simplex = OpenSimplex(seed)

        if torch.is_tensor(t_slice) and t_slice.dim() > 0:
            z = float(t_slice[b])
        else:
            z = float(t_slice)

        img_noise = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        frequency = base_frequency

        for _ in range(octaves):
            noise_fn = np.vectorize(simplex.noise3)
            octave_noise = noise_fn(xx * frequency, yy * frequency, z).astype(np.float32)
            img_noise += amplitude * octave_noise
            amplitude *= decay
            frequency *= 2

        img_noise = (img_noise - img_noise.mean()) / (img_noise.std() + 1e-8)
        noise_batch.append(img_noise)

    noise_batch = np.stack(noise_batch, axis=0)
    noise_batch = torch.tensor(noise_batch, dtype=torch.float32, device=device)
    return noise_batch.unsqueeze(1)

def load_x0(uploaded_file, size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(uploaded_file)
    return transform(img).unsqueeze(0)

st.title("Simplex Noise Visualizer")

file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
size = st.slider("Resize", 64, 256, 128, step=32)
octaves = st.slider("Octaves", 1, 10, 6)
exp = st.slider("Base Frequency (2^x)", -10, -3, -5)
base_freq = 2 ** exp
decay = st.slider("Decay", 0.1, 1.0, 0.8, step=0.05)


if file:
    with st.spinner("Noisificating your image"):
        x0 = load_x0(file, size)
        B, C, H, W = x0.shape

        noise = SimplexNoise(B, H, W, octaves = octaves, base_frequency=base_freq, decay=decay, device=x0.device)

        noise = noise / (noise.std() + 1e-8)

        x_noisy = x0 +  noise

        def to_img(x):
            x = (x + 1) / 2
            return x.clamp(0, 1)[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].imshow(to_img(x0), cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(to_img(x_noisy), cmap="gray")
        axes[1].set_title("Noisy")

        for ax in axes:
            ax.axis("off")

        st.pyplot(fig)