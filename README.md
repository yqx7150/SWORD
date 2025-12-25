# SWORD

**Paper**: Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-View CT Reconstruction           

**Authors**: Kai Xu; Shiyu Lu; Bin Huang; Weiwen Wu; Qiegen Liu          
https://ieeexplore.ieee.org/abstract/document/10403850   
IEEE Transactions on Medical Imaging    

The code and the algorithm are for non-comercial use only.
Copyright 2024, School of Information Engineering, Nanchang University.

Diffusion model has emerged as a potential tool to tackle the challenge of sparse-view CT reconstruction, displaying superior performance compared to conventional methods. Nevertheless, these prevailing diffusion models predominantly focus on the sinogram or image domains, which can lead to instability during model training, potentially culminating in convergence towards local minimal solutions. The wavelet transform serves to disentangle image contents and features into distinct frequency-component bands at varying scales, adeptly capturing diverse directional structures. Employing the wavelet transform as a guiding sparsity prior significantly enhances the robustness of diffusion models. In this study, we present an innovative approach named the Stage-by-stage Wavelet Optimization Refinement Diffusion (SWORD) model for sparse-view CT reconstruction. Specifically, we establish a unified mathematical model integrating low-frequency and high-frequency generative models, achieving the solution with an optimization procedure. Furthermore, we perform the low-frequency and high-frequency generative models on wavelet’s decomposed components rather than the original sinogram, ensuring the stability of model training. Our method is rooted in established optimization theory, comprising three distinct stages, including low-frequency generation, high-frequency refinement and domain transform. The experimental results demonstrated that the proposed method outperformed existing state-of-the-art methods both quantitatively and qualitatively.

## The SWORD training process
![fig1](src/fig1.png)

   
## The pipeline of iterative reconstruction procedure in SWORD
![fig2](src/fig2.png)


## Reconstruction images from 60 views using different methods.
![fig3](src/fig3.png)
(a) The reference image, (b) FBP, (c) U-Net, (d) FBPConvNet, (e) patch-based DDPM, (f) GMSD, (g) SWORD. 

## Training
Wavelet-based Full-frequency Diffusion Model (WFDM)
```bash
python main_wavelet.py --config=aapm_sin_ncsnpp_wavelet.py --workdir=exp_wavelet --mode=train --eval_folder=result
```

Wavelet-based High-frequency Diffusion Model (WHDM)
```bash
python main_3h.py --config=aapm_sin_ncsnpp_3h.py --workdir=exp_3h --mode=train --eval_folder=result
```
## Test
```bash
python PCsampling_demo.py
```


## Test Data
In file './Test_CT', 12 sparse-view CT data from AAPM Challenge Data Study.


### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/OSDM/blob/main/All-CT.png" >  </div>   
    
  * Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10233041)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GMSD)

  * One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10506793)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/OSDM)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
    
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Wavelet-improved score-based generative model for medical imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10288274)

  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  

  * Dual-Domain Collaborative Diffusion Sampling for Multi-Source Stationary Computed Tomography Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10577271)   [<font size=5>**[Code]**</font>](https://github.com/lizrzr/DCDS-Dual-domain_Collaborative_Diffusion_Sampling)

  * Low-rank Angular Prior Guided Multi-diffusion Model for Few-shot Low-dose CT Reconstruction     
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10776993)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PHD)

  * Physics-informed DeepCT: Sinogram Wavelet Decomposition Meets Masked Diffusion  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2501.09935)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SWARM)    
                    
  * MSDiff: Multi-Scale Diffusion Model for Ultra-Sparse View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2405.05763)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MSDiff)

  * Ordered-subsets Multi-diffusion Model for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2505.09985)
                          
  * Virtual-mask Informed Prior for Sparse-view Dual-Energy CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2504.07753)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VIP-DECT)

  * Raw_data_generation  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Raw_data_generation)

  * PRO: Projection Domain Synthesis for CT Imaging  [<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2506.13443)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PRO)
       
  * UniSino: Physics-Driven Foundational Model for Universal CT Sinogram Standardization[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2508.17816)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UniSino)

  * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT) 



