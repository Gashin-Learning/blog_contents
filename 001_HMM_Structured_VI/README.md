
# Hidden Morkov Model Structured Variational Inference

```math
p(s_n|s_{n-1}, {\bf A})=\prod_{i = 1}^{K} Cat({\bf s}_n|{\bf A}_{:,i})^{s_{n-1}, i}
p(s_1|\boldsymbol\pi)=Cat(s_1|\boldsymbol\pi)
```

## reference
[My blog: gashin-learning.com](https://gashin-learning.hatenablog.com/entry/2019/08/25/222422)


## notebook
[here](https://github.com/Gashin-Learning/blog_contents/blob/master/001_HMM_Structured_VI/Comparison_between_HMM_structured_VI_and_PMM_VI.ipynb)

## libarary version

```
 numpy 1.16.4
 scipy 1.3.0 
 matplotlib 3.1.1'
```
