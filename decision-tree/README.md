# 决策树
**id3/c4.5**：id3和c4.5都是每次分岔时，都基于不确定性最小（分类地最干净）的原则选择一个特征维度进行分岔，其中不确定性的度量：
- id3：使用的是**信息增益**，也就是分岔后不确定性减少地最多（**熵**减小地最多）；
- c4.5：使用的是**信息增益比**，在信息增益的基础上除去了特征维度规模造成的信息增益（说人话就是，如果分岔太多的话，会分类地更干净，但是会导致树过于复杂，也就是过拟合，所以在这里做出一点惩罚）。

**剪枝（pruning）**：决策树的防止过拟合的手段：
- 预剪枝：就是在所有特征维度展开的信息增益或者信息增益比都不太高的时，不再进行展开，id3和c4.5用的就是这种策略，比较简单；
- 后剪枝：在展开成完整的树之后再剪枝；选择一个最好的剪枝需要使用独立验证集，有非常多的启发式策略。

**cart**：既能做回归又能做分类，既能用于离散变量也能用于连续变量：
- 生成：cart是二叉树，每次分岔都会选择一个分类效果最好的特征维度 $x^{(j)}$，以及来自样本的该维度下分类效果最好的值 $a_{jl}$来对特征空间进行二分（回归和分类有着不同的公式来衡量分类效果）（与kd树非常相似）（不像id3和c4.5那样，cart可能会多次使用相同的特征维度进行二分）。
- 剪枝：很有趣的后剪枝，书上没有说清楚，感性的理解就是，bottom-up地每次尽可能少地剪去一个子树直到剪完，再这个过程中可以得到一串慢慢变小的决策树序列 $\{T_0,T_1,...,T_k\}$，这个过程等效于，限制树规模的罚项 $\alpha$增加，导致最好的树不断减少的过程；最终我们需要通过交叉验证，来选出泛化能力最好的树 $T_k$
