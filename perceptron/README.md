# 感知机

**约束**：要求数据集必须是线性可分的，否则在随机梯度下降中会出现震荡无法收敛（SVM会针对这个严重缺点进行改进）。

$$
\begin{aligned}
w &\leftarrow w +\eta y_i x_i \\
b &\leftarrow b +\eta y_i
\end{aligned}
$$

**对偶形式**：对训练过程中的判别不等式做了变形，训练的参数也发生了变化；使得在更新参数时不需要计算点积，而是事先把点积都打表了。针对特征比较多的数据集，迭代次数也会很多，优化会更明显（而且方便引入核函数，这部分会在SVM中涉及到）。

$$
\begin{aligned}
\alpha_i &\leftarrow \alpha_i +\eta \\
b &\leftarrow b +\eta y_i
\end{aligned}
$$

PS：代码中可视化的部分，理论上只有特征空间为2维的才有较好的视觉效果，因为高维特征空间的超平面，只取两个维度来投影是高度失真的。