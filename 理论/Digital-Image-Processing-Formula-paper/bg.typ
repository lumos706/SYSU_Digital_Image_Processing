#import "@preview/mitex:0.2.4": *

#heading[BG math]

高斯积分 $integral_(-infinity)^(+infinity) e^(-x^2) d x = sqrt(pi)$

傅里叶级数 $f(t) = sum_(n = - infinity)^infinity c(n) e^(j frac(2 pi n, T) t) c(n) = frac(1, T) integral_(-T \/ 2)^(T \/ 2) f(t) e^(-j frac(2 pi n, T) t) d t$

傅里叶变换 $f(t) = integral_(-infinity)^infinity F(mu) e^(j 2 pi mu t) d mu quad F(mu) = integral_(-infinity)^infinity f(t) e^(-j 2 pi mu t) d t$