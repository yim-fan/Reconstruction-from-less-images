import numpy as np
import os
import imageio
from PIL import Image
import json


def load_blender_data(basedir, cimle_dir, num_hypothesis=20, split='train', factor=1, read_offline=True):
    with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
        meta = json.load(fp)

    poses = []
    images = []
    image_paths = []
    fx=0.0
    fy=0.0
    leres_dir = os.path.join(basedir, "train", "leres_cimle", cimle_dir)
    paths = os.listdir(leres_dir)
    all_depth_hypothesis = []
    near = float(meta['near'])
    far = float(meta['far'])
    for i, frame in enumerate(meta['frames']):
        img_path = os.path.abspath(os.path.join(basedir, frame['file_path']))
        poses.append(np.array(frame['transform_matrix']))
        image_paths.append(img_path)

        if read_offline:
            img = imageio.imread(img_path)
            W, H = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((W//factor, H//factor))
            images.append((np.array(img) / 255.).astype(np.float32))
        elif i == 0:
            img = imageio.imread(img_path)
            W, H = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((W//factor, H//factor))
            images.append((np.array(img) / 255.).astype(np.float32))

    poses = np.array(poses).astype(np.float32)
    images = np.array(images).astype(np.float32)

    H, W = images[0].shape[:2]
    frame = meta['frames'][0]
    fx = float(frame['fx'])
    fy = float(frame['fy'])
    focal = 0.5 * (fx + fy)

    if split == 'train':
        for i, frame in enumerate(meta['frames']):
        ############################################    
        #### Load cimle depth maps ####
        ############################################    
        ## For now only for train poses
        # filename = img_path[train_idx[i]]
            img_id = img_path.split("/")[-1].split(".")[0]
            curr_depth_hypotheses = []

            for j in range(num_hypothesis):
                cimle_depth_name = os.path.join(leres_dir, img_id+"_"+str(j)+".npy")
                cimle_depth = np.load(cimle_depth_name).astype(np.float32)

                ## To adhere to the shape of depths
                # cimle_depth = cimle_depth.T ## Buggy version
                cimle_depth = cimle_depth
            
                cimle_depth = np.expand_dims(cimle_depth, -1)
                curr_depth_hypotheses.append(cimle_depth)

            curr_depth_hypotheses = np.array(curr_depth_hypotheses)
            all_depth_hypothesis.append(curr_depth_hypotheses)

    all_depth_hypothesis = np.array(all_depth_hypothesis)

    ### Clamp depth hypothesis to near plane and far plane
    all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)
            

    return images, poses, [H, W, focal], image_paths, all_depth_hypothesis



