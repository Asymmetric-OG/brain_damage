# Diffusion Based Unsupervised Brain Tumor Detection

<img width="1920" height="1080" alt="mri_diff_image" src="https://github.com/user-attachments/assets/da5020ae-1341-4612-bda3-3ffb3de92b2b" />

## A. INTRODUCTION
### A-1. What is a Diffusion Model?
Denoising Diffusion Probablistic Models or DDPMs were developed and are traditionally used for the purpose of Synthetic Image Generation. 
The primary principle behind this mechanism is repeatedly injecting an image ($$x_0$$) with small amounts of Gaussian Noise (Forward diffusion) over many timesteps (say t) to give a noisy image ($$x_t$$)  and training a symmetric encoder-decoder style CNN, called a UNET, to predict the noise that was added.    
  
The forward diffusion proces follows a Markov Chain, which means $$x_t$$ probabilistically depends only on $$x_{t-1}$$. Hence $$x_{t-1}$$ can be derived using $$x_{t}$$ and the predicted noise. To generate a synthetic image pure noise is taken initially (t=1000 generally) and this process is continued stochastically 
(a small amount of noise is added after each step to create a degree of randomness in SIG) to eventually obtain a brand new image ($$X_0$$) which belongs to the same category of images as ($$x_0$$) but is a different synthetically generated image.  
### A-2. Who Asked?
Right. So why do we care?  
  
This entire mechanism is extremely efficient at learning and producing the structure of a particular category of images (Industrial machinery, Medical Images, etc.). 
Therefore if the model is trained only on Normal/Healthy images, it learns the shape, texture, formation of a normal image by repeatedly predicting the noise
at different timesteps. As a result of this a noisy image ($$x_t$$) at a certain timestep t , when denoised through the reverse diffusion process will always tend to reconstruct a normal image.   
  
Finally, when an anomolous image is noised to $$x_t$$ and then subsequently denoised by the model, the result will be an image that is structurally 
identical (ideally) to the anomolous image in all respects, except the anomaly itself. The difference between this reconstructed image and the original image is what
we call as the anomaly map. However a few adjustments in methodolgy do need to be made in order to obtain near-optimal results.

---

## B. MAIN COMPONENTS
### B-1. Simplex Noise
#### B-1-a. Why betray Gaussian Noise?
Gaussian Noise is widely used in synthetic image generation tasks, where the process benefits from randomness and stochasticity. However it proves to be too noisy for our purpose of reconstructing an image structurally near-identical to the original image (Excluding anomaly if present).  

In contrast to Gaussian Noise, where the noise added to each pixel is randomly sampled, Simplex noise uses a deterministic function to calculate the noising of a pixel. The magnitude of noise added to a pixel has a degree of randomness to it, however the distribution of noise among pixels across the image is highly correlated.
#### B-1-b. Math behind Simplex
Let's imagine 2-D space is divided into a grid of the simplest closed figures possible (Simplices) in this dimension - Triangles.  
<br>
<img width="1920" height="800" alt="simplex exp" src="https://github.com/user-attachments/assets/5952a22c-5146-41f3-af4d-b61b4a93dc3f" />
<br>
Point P represents a pixel of an image existing in this 2-D space and every vertex in this grid is assigned a random vector. Noise for pixel P depends only on the vertex vectors of the triangle pixel P is in and its distances from those vertices. The dot product between each vertex vector and distance vector between P and that vertex is calculated. For each vertex its contribution is regularized by a distance factor (Higher the distance lower the contribution). The total summation of these values is the final noise for Pixel P.  
<br>
It can be observed that points near P will lie in the same or adjacent triangles, making the noise added to them mathematically similar. This creates the smoothness in Simplex Noise.
#### B-1-c. Tweaking Hyperparameters

| Hyperparameter | Chosen value | Description |
|----------|----------|----------|
| Base frequency | $2^{-5}$  | Lower frequency results in spread out noise while Higher frequency creates spotty noise. |
| Octaves  | 6 | Number of layers, with each subsequent layer having higher frequencies. |
| Decay  | 0.8 | The magnitude of each subsequent layer is multiplied by the decay factor, so as to make sure Higher frequency layers don't dominate. |

The overall texture of simplex noise is defined by it's `frequency`. For the purpose of this project, where MRI scans are highly textured due to the brain's folds, we stack multiple layers of simplex noise on top of each others. `Octaves` and `Decay` control how these layers are stacked.

To interactively observe how the noise changes with these hyperparameters
```
streamlit run simplex_visualizer.py
```
##

### B-2. Noise Scheduler

The Noise scheduler controls and implements the forward and reverse diffusion processes.  

#### B-2-a. Forward Diffusion

The Noise Scheduler determines the rate at which an image gets corrupted by defining the `beta vs timestep` curve. Beta decides what factor of the Noise will be added at timestep t. Higher the beta, higher is the noise.  

Naturally there are many curves a noise scheduler can follow, but two of the most common ones are:

- Linear
- Cosine

<img width="1920" height="658" alt="noisesched" src="https://github.com/user-attachments/assets/a4cf2058-8bb2-4b6f-8e50-4e8a387981cf" />  

<br>
As we can see the cosine scheduler adds noise more gradually, letting intricate details of the image survive for a larger timestep.  
Hence the use of Cosine Noise Scheduler in this project.<br>
<br>

```
scheduler.forward_diffusion(x0, noise, t)
```
Injects image with `timestep=t` amount of noise.
#### B-2-b. Reverse Diffusion

1. reverse_timestep : Samples $$x_{t-1}$$ from $$x_t$$ and predicted noise. More useful for synthetic image generation as a step by step denoising process is noisy for accurate reconstruction.
```
scheduler.reverse_timestep(xt, noise_pred, t)
```
2. reconstruct : Predicts $$x_0$$ directly from $$x_t$$ given predicted noise. This is the function we use to get the reconstructed image from the noisy image through a one-shot calculation based on the derivation of minimum log-prob loss for P($$x_0$$) from P($$x_t$$).
```
scheduler.reconstruct(xt, noise_pred, t)
```

##

### B-3. U-NET Architecture

<img width="1448" height="800" alt="Unet" src="https://github.com/user-attachments/assets/e56296d0-1ee4-4b4f-ac90-a3af05f1ab8b" />
<br>

U-NET is an encoder-decoder Convolutional Neural Network architecture that specialises in image-to-image translation through a downsampling and upsampling process.
    
1. Encoder : The downsampling part of a U-NET convolves an image into a dense representation which represents its high level features and details.
2. Decoder : The upsampling part of a U-NET takes the dense representation as input along with skip connections from corresponding encoder levels and interpolates it into a high resolution output image  
  
We train the U-NET to take a Noisy image ($$x_t$$) as input, encode it into a dense representation and then decode it into the output, which is the noise that was added to it at timestep t.

---

## C. TRAINING AND INFERENCE

### C-1. Training the U-NET

The model was trained on 2000+ MRI scans of healthy (Non-tumerous) brains across multiple different contrast weightings (T1, T2, Flair, etc).  
#### C-1-a. Image Preprocessing
Slight augmentations were applied to the image to prevent overfitting and ensure the model adjusts to slight variations present in modern MRI-scans.  

```
no_tumor_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(
        degrees=2,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02),
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```
#### C-1-b. Training Hyperparameters
A cosine learning rate scheduler was used to ensure a smooth training process which is critical for sensitive processes such as diffusion.  

```
t_epochs=250
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
total_training_steps = len(diff_tloader) * t_epochs
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=1500,
    num_training_steps=total_training_steps
)
```
#### C-1-c. Training Trajectory

| Epoch | Train_Loss | Val_Loss |
|----------|----------|----------|
|    10      |   0.1041     |   0.0902   |
|    50      |   0.0600     |   0.0707   |
|    100     |   0.0496     |   0.0521   |
|    150     |   0.0393     |   0.0425   |
|    200     |   0.0379     |   0.0355   |
|    250     |   0.0332     |   0.0357   |

##

### C-2. Generating Anomaly Maps

An MRI-Scan of a brain (Healthy/Tumorous) is noised upto a chosen `timestep t`. This noisy image ($$x_t$$) is then passed through the model to obtain predicted noise ($$noise_pred$$). These three values are passed to the `scheduler.reconstruct(x0, noise_pred, t)` function to obtain the reconstructed image. The anomaly map is obtained by finding out the square error between the original image and reconstructed image.
```
anomaly_map = (reconstructed_image - original_image).pow(2)
```
  
For this process to work well, it is extremely important to choose the timestep carefully. A low timestep might not corrupt the tumor at all and a higher timestep might corrupt the natural structure of the brain too much.

---

# D. BIBLIOGRAPHY

Going through the entire process while understanding every step intricately look a lot of reading and watching from these resources.

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. [AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise](https://www.researchgate.net/publication/362897861_AnoDDPM_Anomaly_Detection_with_Denoising_Diffusion_Probabilistic_Models_using_Simplex_Noise)
3. [Diffusion Models | Math Explained](https://youtu.be/HoKDTa5jHvg?si=UkMMEt90aM246jOS)

---
