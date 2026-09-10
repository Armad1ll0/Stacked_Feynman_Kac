# Stacked Feynman-Kac

The following repository contains the code for reproducing the results for the paper: **Stacked Feynman-Kac: A Generalised Method for Within-Timestep Sampling of Intermediate Distributions for Diffusion Models**. 

# Instructions for repeating the experiments

If testing this repo, we suggest running everything with MNIST first with the instructions below as generating all the results may take a while, thus the instructions given are with fewer samples. The full analysis can be done by altering the commands given below. 

## Installation 

```bash
pip install -r requirements.txt
```

## Image Reconstruction  

```bash
python -m experiments.generate_samples --device cuda --only mnist
```

```bash
python -m experiments.run_experiments --force --only_dataset mnist --only_task deblur --only_method tds tds_hmc_refined --sigma_y 0.05
```

### Analysis 

```bash
python -m analysis.image_restoration.compute_lpips
python -m analysis.image_restoration.generate_reconstructions --dataset mnist --index 00 --output figs/mnist_01.pdf --transpose --method-gap 0.1 --task-gap 0.15
python -m analysis.image_restoration.plot_ess
```

## Gaussian Mixture Model Posterior Sampling 

```bash
python -m experiments.gmm_experiments.run_gmm_comparison --sigma_ys 0.1 1.0 3.0 --seeds 0 1 2 3 4 --tag main
```

### Analysis 

```bash
python -m analysis.gmm.plot_results_gmm gmm_results/main_raw.pt
```
 
## Class Condonditional Sampling

```bash
python -m experiments.class_conditional_experiments.train_classifiers --which both 
```

```bash
python -m experiments.class_conditional_experiments.run_class_experiment --samples-per-class 10 --samples-per-run 10 --num-particles 10 --num-steps 100
```

### Analysis 

```bash
python -m analysis.class_conditional.class_cond_plots --root runs/exp --out-dir figs --panel-style bars
```


## Ablation Results

```bash
python -m experiments.run_ablations --ablation wallclock --num_images 20 --device cuda:0
python -m experiments.run_ablations --ablation stop_at_step --stop_values 5 10 25 50 --num_images 20
```

### Analysis 
```bash
python -m analysis.ablation.check_consistency --csv experiments/results/summary_noisy.csv --sigma 0.05 --markdown
```

```bash
python -m analysis.ablation.analyse_ablation
```

