
#import "@preview/mitex:0.2.4": *
#heading[Chapter 6]

== 彩色基础

红,绿,蓝量用X,Y,Z表示,叫三色值

三色系数定义: $x = frac(X, X + Y + Z);...;x+y+z=1$

== 彩色模型

=== RGB

显示器显示,一个颜色有8比特,2^8=256种颜色,全彩色则是24比特图像

#image("./img/rgb.png",height: 5%)
=== CMYK

颜料颜色;CMY(青色、深红、黄色)是RGB的补色;K是黑色,用于调节色彩

RGB->CMY: $mat(C; M; Y) = mat(1; 1; 1) - mat(R; G; B)$

RGB->CMYK:

$mat(delim: #none, K = 1 - max(R comma G comma B); C = frac(1 - R - K, 1 - K); M = frac(1 - G - K, 1 - K); Y = frac(1 - B - K, 1 - K))$

CMY->CMYK: $K=min(C,M,Y)$ K=1,其他=0

k!=1,其他为$C =(C - K) \/(1 - K); M =(M - K) \/(1 - K); Y =(Y - K) \/(1 - K)$

CMYK->CMY: $C = C(1 - K) + K; M = M(1 - K) + K; Y = Y(1 - Y) + K$

=== HSI

h色调(角度),s饱和度(鲜艳程度),i强度(颜色的明暗程度)
#image("./img/hsi.png",height: 5%)

RGB->HSI

$H = 360 - arccos(frac((R - G) +(R - B), 2 sqrt((R - G)^2 +(R - B)(G - B)))) (G<B)$

$H = arccos(frac((R - G) +(R - B), 2 sqrt((R - G)^2 +(R - B)(G - B)))) (G>=B)$

$S = 1 - frac(3, R + G + B) dot.op  min(R, G, B)$

$I = frac(R + G + B, 3)$

HSI->RGB

$1.0^circle.small lt.eq H < 120^circle.small$

$& R = I dot.op (1 + frac(S dot.op  cos(H), cos(60^circle.small - H))) ;
& G = I dot.op (1 + S dot.op (1 - cos(H) \/ cos(60^circle.small - H))) ;
& B = I dot.op (1 - S)$

$2.120^circle.small lt.eq H < 240^circle.small$

$& H' = H - 120^circle.small \
& G = I dot.op (1 + frac(S dot.op cos(H'), cos(60^circle.small - H'))) ;
& B = I dot.op (1 + S dot.op (1 - cos(H') \/ cos(60^circle.small - H'))) ;
& R = I dot.op (1 - S)$


$3.240^circle.small lt.eq H < 360^circle.small$


$& H' = H - 240^circle.small \
& B = I dot.op (1 + frac(S dot.op cos(H'), cos(60^circle.small - H'))) ;
& R = I dot.op (1 + S dot.op (1 - cos(H') \/ cos(60^circle.small - H'))) ;
& G = I dot.op (1 - S)$

=== CIELAB

$L_star = 116 * h(Y / Y_W) - 16\
a_star = 500 * [h(X / X_W) - h(Y / Y_W)]\
b_star = 200 * [h(Y / Y_W) - h(Z / Z_W)]\
\
h(q) = cases(
  q > 0.008856 => (3 / 2) * q^(1/3),
  q <= 0.008856 => 7.787 * q + 16 / 116
)$

L表示亮度，范围从0（黑色）到100（白色）。
a表示从绿色到红色的轴。
b表示从蓝色到黄色的轴。
h(q)是一个辅助函数，用于处理非线性变换。

== 假彩色
[0,L-1]灰度级别,分为P+1个区间,$I_1, I_2, dots.c, I_(P + 1)$,属于某个区间就赋值一个彩色

$"若" f(x, y) in I_k, "则令" f(x, y) = c_k$

也可以设置$f_R,f_G,f_B$把灰度映射为不同通道的颜色

== 彩色变换
//6.5
$s_i = T_i (r_i), quad i = 1, 2, dots.c, n$

提高亮度:RGB三个分量乘以常数k;CMY求线性变化$s_i = k r_i +(1 - k), quad i = 1, 2, 3$;CMYK只需改变第四个分量(K)$s_i = k r_i +(1 - k), quad i = 4$

=== 彩色分层

突出图像中某个特定的彩色范围，有助于将目标从周围分离出来

感兴趣的颜色被宽度为W、中心在原型(即平均)颜色并具有分量$a_j$的立方体(n>3时为超立方体)包围，

$s_i = cases(0 . 5 comma\, quad [ lr(|r_j - a_j|) > W \/ 2 ]_(1 lt.eq j lt.eq n), r_i comma\, quad "其他",) quad i = 1, 2, dots.c, n$

用一个球体来规定感兴趣的颜色时

$s_i = cases(0 . 5 comma\, quad sum_(j = 1)^n (r_j - a_j)^2 > R_0^2, r_i comma\, quad "其他",) quad i = 1, 2, dots.c, n$

== 平滑和锐化

=== 平滑
$overline(c)(x, y) = mat(frac(1, K) sum_((s comma t) in S_(s y)) R(s comma t);; frac(1, K) sum_((s comma t) in S_(s y)) G(s comma t);; frac(1, K) sum_((s comma t) in S_(s y)) B(s comma t))$

=== 锐化

$nabla^2 mat(delim: #none, c(x comma y)) = mat(nabla^2 R(x comma y);; nabla^2 G(x comma y);; nabla^2 B(x comma y))$

== 分割图像

HSI:用饱和度(S),大于某个阈值分割

RGB: 令z表示RGB空间中的任意一点,RGB向量a来表示平均颜色

欧氏距离为 $D(z, a) & =|z - a| \ & = [(z - a)^upright(T) (z - a) ]^frac(1, 2) \ & = [(z_R - a_R)^2 +(z_G - a_G)^2 +(z_B - a_B)^2 ]^frac(1, 2)$

$D(z,a)≤D_0$的点的轨迹是半径为$D_0$的一个实心球体

马哈拉诺比斯距离 $D(z, a) = [(z - a)^upright(T) C^(-1)(z - a) ]^frac(1, 2)$

$D(z,a)≤D_0$的点的轨迹是半径为$D_0$的一个实心三维椭球体

两个方法都计算代价也很高昂,一般用边界盒关于a居中，它沿各坐标轴的长度与样本沿坐标轴的标准差成比例

//彩色边缘检测
//似乎比较难?

// == 噪声

// == 压缩