
## 处理序列对齐
if fea_stack_a.shape[1] == fea_stack_v.shape[1]:
    pass
elif fea_stack_a.shape[1] < fea_stack_v.shape[1]:
    s = fea_stack_v.shape[1] - fea_stack_a.shape[1]
    p1d = [0, 0, 0, s]
    fea_stack_a = F.pad(fea_stack_a, p1d, value=train_loader.dataset.pad_idx)
elif fea_stack_a.shape[1] > fea_stack_v.shape[1]:
    s = fea_stack_a.shape[1] - fea_stack_v.shape[1]
    p1d = [0, 0, 0, s]
    fea_stack_v = F.pad(fea_stack_v, p1d, value=train_loader.dataset.pad_idx)