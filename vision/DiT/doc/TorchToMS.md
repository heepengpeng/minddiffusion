# PyTorch 转 MindSpore

## 1. 预训练模型转化
``python
for k, v in state_dict.items():
        if 'embedding_table' in k:
            k = k.replace('weight', 'embedding_table')
        ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
``
## 2. API转换

## 2.1 运算符类
exp，tanh，log，where，sqrt,BatchMatMul
Einsum（MindSpore只支持GPU）

## 2.2 生成随机分布
StandardNormal(需要给定随机种子)
XavierUniform

## 2.3 数据变换 
split，cat，Concat，cast（使用体验较差），stack,AllGather，expand_dims
