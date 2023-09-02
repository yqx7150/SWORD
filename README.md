# SWORD

**Paper**: Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-View CT Reconstruction           

**Authors**: Kai Xu; Shiyu Lu; Bin Huang; Weiwen Wu; Qiegen Liu          


## Training
wavelet-based full-frequency diffusion model (WFDM)
```bash
python main_wavelet.py --config=aapm_sin_ncsnpp_wavelet.py --workdir=exp_wavelet --mode=train --eval_folder=result
```

wavelet-based high-frequency diffusion model (WHDM)
```bash
python main_3h.py --config=aapm_sin_ncsnpp_3h.py --workdir=exp_3h --mode=train --eval_folder=result
```
## Test
```bash
python PCsampling_demo.py
```


## Test Data
In file './Test_CT', 12 sparse-view CT data from AAPM Challenge Data Study.
