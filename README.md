# weight_pruning
Pruning weights on convolutional neural network by pytorch.
基于幅度的权值剪枝方法：

将权值取绝对值，与设定的 threshhold 值进行比较，低于门限的权值被置零。

对每个被选中做剪枝的层增加一个二进制掩模（mask）变量，形状和该层的权值张量形状完全相同。
该掩模决定了哪些权值参与前向计算。
对幅度小于一定门限的权值将其对应掩模值设为 0。
反向传播梯度也经过掩模，被屏蔽的权值（mask 为 0）在反向传播步骤中无法获得更新量。