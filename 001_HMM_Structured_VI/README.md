
# Hidden Morkov Model Structured Variational Inference


## model and inference
<dl>
  <dt>system model</dt>
  <img src="https://latex.codecogs.com/gif.latex?p(s_n|s_{n-1},&space;{\bf&space;A})=\prod_{i&space;=&space;1}^{K}&space;Cat({\bf&space;s}_n|{\bf&space;A}_{:,i})^{s_{n-1},&space;i}">
</dl>
<dl>
  <dt>observation model</dt>
  <img src="https://latex.codecogs.com/gif.latex?p(x_n|s_n)=\prod_{k&space;=&space;1}^{K}&space;Poi(x_n|\lambda_k)^{s_n,&space;k}$$\(\lambda_k\)">
</dl>
<dl>
  <dt>structured variational inference</dt>
 <img src="https://latex.codecogs.com/gif.latex?p({\bf&space;S},&space;\boldsymbol\lambda,&space;{\bf&space;A},&space;\boldsymbol\pi)&space;\approx&space;q({\bf&space;S})q(\boldsymbol\lambda,&space;{\bf&space;A},&space;\boldsymbol\pi)">
</dl>



## reference
My blog: [gashin-learning.hatenablog.com/](https://gashin-learning.hatenablog.com/entry/2019/08/25/222422)


## notebook
[here](https://github.com/Gashin-Learning/blog_contents/blob/master/001_HMM_Structured_VI/Comparison_between_HMM_structured_VI_and_PMM_VI.ipynb)

## libarary version

```
 numpy 1.16.4
 scipy 1.3.0 
 matplotlib 3.1.1'
```
