
## 关于titan的lr和grad shape

lr是per token的,但是grad是per chunk,如何对应
	广播,实际上lr是出现在loss weight上实现的

核心code

```python 
# neural_memory.py
def forward_and_loss(params, inputs, loss_weights, target):
    # inputs (keys): [batch*heads*num_chunks, chunk_size*num_updates, dim] 
    # loss_weights (lr): [batch*heads*num_chunks, chunk_size*num_updates] num_updates这里可以理解为多头,应该是lucidrains自己发挥的
    
    pred = functional_call(self.memory_model, params, inputs)
    # pred: [batch*heads*num_chunks, chunk_size*num_updates, dim]
    
    loss = (pred - target).pow(2).mean(dim=-1)
    # loss: [batch*heads*num_chunks, chunk_size*num_updates]
    
    weighted_loss = loss * loss_weights  
    # weighted_loss: [batch*heads*num_chunks, chunk_size*num_updates]
    
    # 1. 关键点: 这里会把chunk_size*num_updates这个维度sum掉
    return weighted_loss.sum()  
    # 返回标量: []
```



## titan-muon:
motivation:

近期一定会有人把adam做到test time training(试过,不太work,猜想momentum沿着chunk更新受num chunk影响)
所以用muon(https://github.com/KellerJordan/modded-nanogpt ## speedrun nanogpt, 唯一一个干掉adamw的optimizer)做grad refine,本质上是对梯度做一定程度的正交化,参考(https://spaces.ac.cn/archives/10592) ##苏剑林解析
grad = zeropower_via_newtonschulz5(grad) 


```python 
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    使用 Newton-Schulz 迭代计算矩阵的零次方/正对化。
    支持批次处理（batch processing）。
    
    参数:
    G: shape为(batch_size, m, n)的3D张量
    steps: 迭代步数
    eps: 数值稳定性的小量
    
    返回:
    shape为(batch_size, m, n)的3D张量
    """
    assert len(G.shape) == 3
    batch_size, m, n = G.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # 转换为bfloat16并归一化
    # X = G.bfloat16()
    X = G
    X = X / (X.norm(dim=(1,2), keepdim=True) + eps)
    
    # 处理非方阵情况
    transpose_needed = m > n
    if transpose_needed:
        X = X.transpose(1, 2)
    
    # Newton-Schulz 迭代
    for _ in range(steps):
        # bmm用于批次矩阵乘法
        A = X @ X.transpose(1, 2)  # shape: (batch_size, m, m)
        B = b * A + c * (A @ A)    # shape: (batch_size, m, m)
        X = a * X + B @ X          # shape: (batch_size, m, n)
    
    # 如果需要，转置回原始形状
    if transpose_needed:
        X = X.transpose(1, 2)
        
    return X
```






