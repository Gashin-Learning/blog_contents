
# Hidden Morkov Model Structured Variational Inference

https://gashin-learning.hatenablog.com/entry/2019/08/25/222422

[System Model]
p(sn|sn−1,A)=∏i=1KCat(sn|A:,i)sn−1,i
p(s1|π)=Cat(s1|π)

[Observation Model]
p(xn|sn)=∏k=1KPoi(xn|λk)sn,k

$$p({\bf S}, \boldsymbol\lambda, {\bf A}, \boldsymbol\pi)  \approx q({\bf S})q(\boldsymbol\lambda, {\bf A}, \boldsymbol\pi)$$
p(S,λ,A,π)≈q(S)q(λ,A,π)
