from .scannet_dataset import ScanNetDataset, convert_depth_completion_scaling_to_m, convert_m_to_depth_completion_scaling, \
    get_pretrained_normalize, resize_sparse_depth, TaskonomyDataset, convert_depth_completion_scaling_to_m_taskonomy, convert_m_to_depth_completion_scaling_taskonomy
from .load_scene import load_scene, load_scene_scannet, load_scene_processed, load_scene_nogt
from .dataset_sampling import create_random_subsets

from .dataset import RINDataset
from torch.utils.data import DataLoader


def get_traindataset(args):
    return RINDataset(args, mode='train')


def get_trainloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


def get_testdataset(args):
    return RINDataset(args, mode='test')


def get_testloader(dataset, args):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def get_dataset(args, mode):
    if mode == 'train':
        return get_traindataset(args)
    elif mode == 'test':
        return get_testdataset(args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))


def get_loader(dataset, args, mode):
    if mode == 'train':
        return get_trainloader(dataset, args)
    elif mode == 'test':
        return get_testloader(dataset, args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))
