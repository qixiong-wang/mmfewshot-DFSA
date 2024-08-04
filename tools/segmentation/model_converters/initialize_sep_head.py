import torch

def convert_checkpoint(old_checkpoint_path, new_checkpoint_path):
    # 加载旧的checkpoint
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['state_dict']

    # 新的state_dict
    new_state_dict = {}

    # 处理backbone
    for key, value in model_state_dict.items():
        if key.startswith('backbone'):
            new_key_base = key.replace('backbone', 'backbone_base', 1)
            new_key_novel = key.replace('backbone', 'backbone_novel', 1)
            new_state_dict[new_key_base] = value
            new_state_dict[new_key_novel] = value

        # 处理neck
        elif key.startswith('neck'):
            new_key_base = key.replace('neck', 'neck_base', 1)
            new_key_novel = key.replace('neck', 'neck_novel', 1)
            new_state_dict[new_key_base] = value
            new_state_dict[new_key_novel] = value

        # 处理decode_head
        elif key.startswith('decode_head'):
            new_key_base = key.replace('decode_head', 'decode_head_base', 1)
            new_key_novel = key.replace('decode_head', 'decode_head_novel', 1)
            new_state_dict[new_key_base] = value
            new_state_dict[new_key_novel] = value

        # 保留其他未修改的键值
        else:
            new_state_dict[key] = value

    # 保存新的checkpoint
    new_checkpoint = {'state_dict': new_state_dict}
    torch.save(new_checkpoint, new_checkpoint_path)
    print(f"新的checkpoint已保存到 {new_checkpoint_path}")

# 使用示例
original_checkpoint_path = 'work_dirs/isaid/base_training/split2/r101_fpn_fsd_isaid-split2_base-training-0/iter_80000.pth'
new_checkpoint_path = 'work_dirs/isaid/base_training/split2/r101_fpn_fsd_isaid-split2_base-training-0/iter_80000_novel.pth'
convert_checkpoint(original_checkpoint_path, new_checkpoint_path)