# SWORD

**Paper**: Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-View CT Reconstruction           

**Authors**: Kai Xu; Shiyu Lu; Bin Huang; Weiwen Wu; Qiegen Liu          
https://ieeexplore.ieee.org/abstract/document/10403850   
IEEE Transactions on Medical Imaging    

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



## Other Related Projects
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
   
  * Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10233041)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GMSD)

  * One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10506793)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/OSDM)

  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  
