
# Latent Dirichlet Allocation topic inference by Particle Filter(Sequential Monte Carlo)

<img src=https://cdn-ak.f.st-hatena.com/images/fotolife/g/gashin_learning/20191103/20191103165515.png>
<img src=https://cdn-ak.f.st-hatena.com/images/fotolife/g/gashin_learning/20191103/20191103162031.png>

<img src=https://latex.codecogs.com/gif.latex?p(z_{d,i}|w_{d,i},&space;{\bf{z}}^{(d,&space;i-1)},&space;{\bf{w}}^{(d,&space;i-1)},&space;\alpha,&space;\beta)\propto&space;\frac{{n_{k,v}}^{(d,{i-1})}&plus;\beta_v}{\sum_{v}({n_{k,v}}^{(d,{i-1})}&plus;\beta_v)}\frac{{n_{d,k}}^{(d,{i-1})}&plus;\alpha_k}{\sum_{k}({n_{d,k}}^{(d,{i-1})}&plus;\alpha_k)}>
## reference
- [2009;Canini]Online Inference of Topics with Latent Dirichlet Allocation(http://proceedings.mlr.press/v5/canini09a.html)
- My blog: [gashin-learning.hatenablog.com/](https://gashin-learning.hatenablog.com/entry/2019/08/28/154500)


## notebook
[here](https://github.com/Gashin-Learning/blog_contents/blob/master/003_LDA_Particle_Filter/LDA_Particle_filter_experiment_note.ipynb)

## libarary version

```
 Python 3.6
 numpy 1.16.4
 numba 0.45.1
 matplotlib 3.1.1
 sklearn 0.21.2
 wordcloud 1.5.0
```


