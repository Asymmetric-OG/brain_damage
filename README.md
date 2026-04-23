# Diffusion Based Unsupervised Brain Tumor Detection

<img width="1292" height="1080" alt="mri_diff_image" src="https://github.com/user-attachments/assets/bf141a53-a27e-4ce2-b6dd-029a10f909d6" />

---

## A. INTRODUCTION
### A-1. What is a Diffusion Model?
Denoising Diffusion Probablistic Models or DDPMs were developed and are traditionally used for the purpose of Synthetic Image Generation. 
The primary principle behind this mechanism is repeatedly injecting an image (x_0) with small amounts of Gaussian Noise (Forward diffusion) over many timesteps 
(say t) to give a noisy image (x_t) and training a symmetric encoder-decoder style CNN, called a UNET, to predict the noise that was added.  
  
The forward diffusion proces follows a Markov Chain, which means x_t probabilistically depends only on x_t-1. Hence x_t-1 can be derived using x_t and the predicted noise. 
To generate a synthetic image pure noise is taken initially (t=1000 generally) and this process is continued stochastically 
(a small amount of noise is added after each step to create a degree of randomness in SIG) to eventually obtain a brand new image X_0 which belongs to the same
category of images as x_0 but is a different synthetically generated image.
### A-2. Who Asked?
Right. So why do we care?  
  
This entire mechanism is extremely efficient at learning and producing the structure of a particular category of images (Industrial machinery, Medical Images, etc.). 
Therefore if the model is trained only on Normal/Healthy images, it learns the shape, texture, formation of a normal image by repeatedly predicting the noise
at different timesteps. As a result of this a noisy image (x_t) at a certain timestep t , when denoised through the reverse diffusion process will always tend to 
reconstruct a normal image.   
  
Finally, when an anomolous image is noised to x_t and then subsequently denoised by the model, the result will be an image that is structurally 
identical (ideally) to the anomolous image in all respects, except the anomaly itself. The difference between this reconstructed image and the original image is what
we call as the anomaly map. However a few adjustments in methodolgy do need to be made in order to obtain near-optimal results.

