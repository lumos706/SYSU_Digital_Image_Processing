
#import "@preview/mitex:0.2.4": *
#import "@preview/tablex:0.0.6": tablex, hlinex
#import "@preview/tablem:0.1.0": tablem

#heading[Chapter 10]

== 背景知识

差分: 前向 $frac(partial f (x), partial x) = f (x + 1) - f (x)$  后向 $frac(partial f (x), partial x) = f (x) - f (x - 1)$  中值 $frac(partial f (x), partial x) = frac(f (x + 1) - f (x - 1), 2)$  二阶 $frac(partial^2 f (x), partial x^2) = f (x + 1) - 2 f (x) + f (x - 1)$

(1)一阶导产生粗边缘；(2)二阶导对精细细节(如细线、孤立点和噪声)有更强的响应；(3)二阶导在灰度斜坡和台阶过渡处会产生双边缘响应；(4)二阶导的符号可用于确定边缘的过渡是从亮到暗(正)还是从暗到亮(负)。

== 孤立点检测

拉普拉斯  $nabla^2 f (x , y) = frac(partial^2 f, partial x^2) + frac(partial^2 f, partial y^2)  = f (x + 1 , y) + f (x - 1 , y) + f (x , y + 1) + f (x , y - 1) - 4 f (x , y)$

超过阈值T的标记 $g (x , y) = cases(delim: "{", 1  "," lr(|Z (x , y)|) > T, 0 "," upright("其他") & )$ $nabla^2 f=Z$

== 直线检测

#image("./img/zxjc.png",height: 5%)

== 边缘检测

梯度 $nabla f (x , y) equiv upright(g r a d) [f (x , y)] equiv mat(delim: "[", g_x (x , y); g_y (x , y)) = mat(delim: "[", frac(partial f (x , y), partial x); frac(partial f (x , y), partial y))$

欧几里得向量范数 $M (x , y) = ||nabla f (x , y)|| = sqrt(g_x^2 (x , y) + g_y^2 (x , y))$ 绝对值来近似梯度幅度：$ M (x , y) approx lr(|g_x|) + lr(|g_y|) $

梯度方向(垂直边缘) $alpha (x , y) = arctan [frac(g_y (x , y), g_x (x , y))]$

$ mat(
  z_1, z_2, z_3;
  z_4, z_5, z_6;
  z_7, z_8, z_9
) $

罗伯算子   $g_x = frac(partial f, partial x) = (z_9 - z_5)$     $g_y = frac(partial f, partial y) = (z_8 - z_6)$

Prewitt算子   $g_x = frac(partial f, partial x) = (z_7 + z_8 + z_9) - (z_1 + z_2 + z_3)$     $g_y = frac(partial f, partial y) = (z_3 + z_6 + z_9) - (z_1 + z_4 + z_7)$

Sobel 算子  $g_x = frac(partial f, partial x) = (z_7 + 2 z_8 + z_9) - (z_1 + 2 z_2 + z_3)$     $g_y = frac(partial f, partial y) = (z_3 + 2 z_6 + z_9) - (z_1 + 2 z_4 + z_7)$

二维高斯函数， $G (x , y) = upright(e)^(- frac(x^2 + y^2, 2 sigma^2))$

高斯拉普拉斯(LoG)函数:  $nabla^2 G (x , y) = (frac(x^2 + y^2 - 2 sigma^2, sigma^4)) upright(e)^(- frac(x^2 + y^2, 2 sigma^2))$

Marr-Hildreth算法  $g (x , y) = [nabla^2 G (x , y)] star.filled f (x , y) = nabla^2 [G (x , y) star.op f (x , y)]$  寻找g(x,y)的过零点来确定f(x,y)中边缘的位置

高斯差分(DoG)来近似式的LoG函数  $D_G (x , y) = frac(1, 2 pi sigma_1^2) upright(e)^(- frac(x^2 + y^2, 2 sigma_1^2)) - frac(1, 2 pi sigma_2^2) upright(e)^(- frac(x^2 + y^2, 2 sigma_2^2))$

// 坎尼 没有公式,跳过

== 连接边缘点

满足条件则连接 $mat(delim: "|", M (s , t) - M (x , y)) lt.eq E$ $mat(delim: "|", alpha (s , t) - alpha (x , y)) lt.eq A$

霍夫变换  $x cos theta + y sin theta = rho$

== 阈值处理

多分类   $g (x , y) = cases(delim: "{", a ", " f (x , y) > T_2, b "," T_1 < f (x , y) lt.eq T_2, c "," f (x , y) lt.eq T_1 & )$

基本的全局阈值化
1. 为全局阈值$T$选择一个初始估计值。
2. 在 $g (x , y) = cases(delim: "{", 1 "," f (x , y) > T, 0 "," f (x , y) lt.eq T & )$中用$T$分割图像。这将产生两组像素：由灰度值大于$T$的所有像素组成的$G_1$,由所有小于等于$T$的像素组成的$G_2$
3. 对 $G_1$ 和 $G_2$中的像素分别计算平均灰度值(均值)$m_1$和 $m_2$
4. 在$m_1$和 $m_2$之间计算一个新的阈值： $T=(m_1+m_2)/2$
5. 重复步骤 2 到步骤 4,直到连续迭代中的两个$T$值间的差小于某个预定义的值$Delta T$为止。

OSTU方法   $n_i$ 表示灰度级i的像素数, $M*N= sum_(i=0)^(L-1) n_i;p_i=n_i/(M N);sum_(i = 0)^(L - 1) p_i = 1 , quad p_i gt.eq 0$

分为两类 $c_1,c_2$  累计概率 $P_1 (k) = sum_(i = 0)^k p_i;P_2 (k) = sum_(i = k + 1)^(L - 1) p_i = 1 - P_1 (k)$  平均灰度  $m_1 (k) = frac(1, P_1 (k)) sum_(i = 0)^k i p_i;m_2 (k) = frac(1, P_2 (k)) sum_(i = k + 1)^(L - 1) i p_i$  k级累计灰度 $m (k) = sum_(i = 0)^k i p_i$  整个图像平均灰度  $m_G = sum_(i = 0)^(L - 1) i p_i$

约束条件 $P_1 m_1 + P_2 m_2 = m_G;P_1 + P_2 = 1$

全局方差  $sigma_G^2 = sum_(i = 0)^(L - 1) (i - m_G)^2 p_i$  类间方差  $sigma_B^2 = P_1 (m_1 - m_G)^2 + P_2 (m_2 - m_G)^2 = P_1 P_2 (m_1 - m_2)^2 = frac((m_G P_1 - m)^2, P_1 (1 - P_1))$  (选择k最大化 $sigma_B^2$ )

也可以多个阈值约束 $sigma_B^2 (k_1^(\*) , k_2^(\*) , dots.h.c , k_(K - 1)^(\*)) = max_(0 < k_1 < k_2 < dots.h.c k_K < L - 1) sigma_B^2 (k_1 , k_2 , dots.h.c , k_(K - 1))$

//P554
== 区域生长 分离 聚合

区域生长
//p565
1. *种子选择*：选择一组“种子”点，这些种子点通常是具有某些特定属性的像素，如灰度或颜色范围。种子点的选择可以根据问题的性质或图像的特性来确定。

2. *相似性准则*：定义一个相似性准则，用于判断邻域像素是否应被添加到当前区域。相似性准则可以基于灰度、颜色、纹理等属性。

3. *区域扩展*：从种子点开始，将满足相似性准则的邻域像素逐步添加到当前区域中。这个过程会不断重复，直到没有更多的像素满足加入准则。

4. *连通性考虑*：在区域生长过程中，必须考虑像素的连通性，以确保生成的区域是连通的。通常使用8连通或4连通来定义邻域。

5. *停止规则*：定义一个停止规则，当没有更多的像素满足加入准则时，区域生长过程停止。

6. *区域标记*：使用不同的标记（如整数或字母）来标识每个生成的区域，形成分割后的图像。

分离聚合
//p567
1. *初始分割*：将图像初步划分为一组不相交的区域（如基于像素的颜色、灰度值等），形成初始区域。这些区域可以用细网格单元表示。

2. *分离规则*：
   - 根据定义的判别准则（如区域的均值、方差、纹理等特性），对某一特定区域 $R$ 判断其是否满足某些属性。如果不满足，则将其细分为更小的不相交区域。
   - 例如，可以将 $Q(R)="FALSE"$ 的任何区域划分为4个子区域。

3. *聚合规则*：
   - 如果满足某些逻辑条件（如两个相邻区域的属性接近，满足 $Q(R_i union R_j) = "TRUE"$），则将这些区域进行合并。
   - 通过不断聚合区域，减少过度分割的可能性。

4. *停止条件*：
   - 当区域无法进一步分割或聚合时，停止操作。
   - 最终的分割结果应满足所有区域均符合准则。

5. *应用示例*：
   - 结合区域的统计特性（如均值 $m_R$ 和标准差 $sigma_R$）和用户定义的阈值范围，可以定义规则 $Q(R)$（例如：$sigma_R > a \, "AND" \, m_R < b$）。
== 分水岭变换

//P585
1. 梯度图像：，算法使用图像的梯度图像 $g(x,y)$，其中包含多个区域极小值 $M_{1}, M_{2},  M_{g}$。这些极小值对应于图像中的局部低谷。

2. 汇水盆地：每个区域极小值 $M_{i}$ 都有一个与之相关联的汇水盆地 $C(M_i)$，这些汇水盆地中的点形成一个连通分量。

3. 淹没过程：算法通过模拟水位从最小值 $min$ 逐渐上升到最大值 $max$ 的过程来分割图像。在每个水位 $n$，集合 $T[n]$ 包含所有灰度值小于 $n$ 的点。

4. 二值图像：在每个水位 $n$，$T[n]$ 可以被视为一幅二值图像，其中黑点表示位于平面 $g(x,y) = n$ 下方的点。

5. 汇水盆地分割：随着水位上升，算法通过比较当前水位 $n$ 的连通分量与前一水位 $n-1$ 的汇水盆地，来确定是否需要构建水坝以防止不同汇水盆地的水流溢出。

6. 水坝构建：当水位上升到某个点时，如果发现有多个汇水盆地的水流可能溢出，算法会在这些汇水盆地之间构建水坝（即分割线），以阻止水流混合。

缺点:受噪声影响大;容易过度分割

