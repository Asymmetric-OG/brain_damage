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
<img width="1920" height="800" alt="simplex exp" src="https://github.com/user-attachments/assets/5952a22c-5146-41f3-af4d-b61b4a93dc3f" />
<br>
Point P represents a pixel of an image existing in this 2-D space and every vertex in this grid is assigned a random vector. Noise for pixel P depends only on the vertex vectors of the triangle pixel P is in and its distances from those vertices. The dot product between each vertex vector and distance vector between P and that vertex is calculated. For each vertex its contribution is regularized by a distance factor (Higher the distance lower the contribution). The total summation of these values is the final noise for Pixel P.  
<br>
<br>
It can be observed that points near P will lie in the same or adjacent triangles, making the noise added to them mathematically similar. This creates the smoothness in Simplex Noise.
#### B-1-c. Tweaking Hyperparameters
The overall texture of simplex noise is defined by it's `frequency`. For the purpose of this project, where MRI scans are highly textured due to the brain's folds, we stack multiple layers of simplex noise on top of each others. `Octaves` and `Decay control` how these layers are stacked.

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
Hence the use of Cosine Noise Scheduler in this project.

#### B-2-b. Reverse Diffusion



