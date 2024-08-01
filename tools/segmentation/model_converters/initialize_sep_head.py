import torch

def convert_checkpoint(original_checkpoint_path, new_checkpoint_path):
    # 加载原始checkpoint
    checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
    new_state_dict = {}

    for key, value in checkpoint['state_dict'].items():
        new_state_dict[key] = value
        
        # 处理neck和decode_head的novel分支
        if 'neck' in key or 'decode_head' in key:
            parts = key.split('.')
            if 'neck' in key:
                # 找到第二级名称并替换
                parts[1] = parts[1] + '_novel'
                new_key = '.'.join(parts)
                new_state_dict[new_key] = value
            elif 'decode_head' in key:
                # 找到第二级名称并替换
                parts[1] = parts[1] + '_novel'
                new_key = '.'.join(parts)
                new_state_dict[new_key] = value

    # 创建新的checkpoint
    new_checkpoint = {
        'state_dict': new_state_dict,
        # 复制其他可能的元信息，例如optimizer等
        'meta': checkpoint.get('meta', {}),
        'optimizer': checkpoint.get('optimizer', None),
        'scheduler': checkpoint.get('scheduler', None),
    }

    # 保存新的checkpoint
    torch.save(new_checkpoint, new_checkpoint_path)
    print(f'New checkpoint saved to {new_checkpoint_path}')

# 使用示例
original_checkpoint_path = 'work_dirs/nwpu/base_training/tfa_r101_fpn_nwpu-split1_base-training_resize-bs8-s4-0/iter_20000.pth'
new_checkpoint_path = 'work_dirs/nwpu/base_training/tfa_r101_fpn_nwpu-split1_base-training_resize-bs8-s4-0/iter_20000_sep_novel.pth'
convert_checkpoint(original_checkpoint_path, new_checkpoint_path)