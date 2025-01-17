import sys
from pathlib import Path

# sys.path.append("/data/sahil/sfm/accelerated_features")
sys.path.append(str(Path(__file__).parent / "../third_party/accelerated_features"))

from modules.xfeat import XFeat

import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import h5py
import torch
from tqdm import tqdm

from . import logger
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .ransac import *  # Assume RANSAC classes and 'ransac' object are defined
import numpy as np
import time
from scipy.spatial import cKDTree
import faiss
faiss.omp_set_num_threads(8)  # Set to the number of available cores

res = faiss.StandardGpuResources()

def cam_from_img_vectorized(params, points):
    PI = 4 * math.atan(1)
    c1, c2 = params[1], params[2]
    theta = (points[:, 0] - c1) * PI / c1
    phi = (points[:, 1] - c2) * PI / (2 * c2)
    u = np.cos(theta) * np.cos(phi)
    v = np.sin(phi)
    w = np.sin(theta) * np.cos(phi)
    return np.column_stack((u, v, w))

# params = [0, 1920/2, 960 / 2]

g8p = EightPointAlgorithmGeneralGeometry()
ransac = RANSAC_8PA()

faces_order = ["f", "r", "b", "l", "u","d"]  # Must match how features are stored


def cubemap_to_equirectangular_uv(face, x, y, cubemap_size=1920, eq_width=7680, eq_height=3840):
    x = (x / cubemap_size) * 2 - 1
    y = (y / cubemap_size) * 2 - 1
    if face == 'F':
        vec = [1, x, -y]
    elif face == 'R':
        vec = [-x, 1, -y]
    elif face == 'B':
        vec = [-1, -x, -y]
    elif face == 'L':
        vec = [x, -1, -y]
    elif face == 'U':
        vec = [y, x, 1]
    elif face == 'D':
        vec = [-y, x, -1]
    vec = np.array(vec) / np.linalg.norm(vec)
    theta = np.arctan2(vec[1], vec[0])
    phi = -np.arcsin(vec[2])
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    return u * eq_width, v * eq_height

def cubemap_to_equirectangular_uv_batch(face, x, y, cubemap_size=1920, eq_width=7680, eq_height=3840):
    # Convert torch.Tensor to numpy.ndarray if necessary
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(cubemap_size, torch.Tensor):
        cubemap_size = cubemap_size.item()
    if isinstance(eq_width, torch.Tensor):
        eq_width = eq_width.item()
    if isinstance(eq_height, torch.Tensor):
        eq_height = eq_height.item()
    
    # Normalize x and y to [-1, 1]
    x = (x / cubemap_size) * 2 - 1
    y = (y / cubemap_size) * 2 - 1
    
    # Determine the direction vector based on the face
    if face == 'F':
        vec = np.stack([np.ones_like(x), x, -y], axis=1)
    elif face == 'R':
        vec = np.stack([-x, np.ones_like(x), -y], axis=1)
    elif face == 'B':
        vec = np.stack([-np.ones_like(x), -x, -y], axis=1)
    elif face == 'L':
        vec = np.stack([x, -np.ones_like(x), -y], axis=1)
    elif face == 'U':
        vec = np.stack([y, x, np.ones_like(x)], axis=1)
    elif face == 'D':
        vec = np.stack([-y, x, -np.ones_like(x)], axis=1)
    else:
        raise ValueError(f"Unknown face: {face}")
    
    # Normalize the vectors
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    
    # Compute theta and phi
    theta = np.arctan2(vec[:, 1], vec[:, 0])
    phi = -np.arcsin(vec[:, 2])
    
    # Convert to equirectangular coordinates
    u = (theta + np.pi) / (2 * np.pi) * eq_width
    v = (phi + np.pi / 2) / np.pi * eq_height
    
    return np.stack([u, v], axis=1)


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    try:
        pair, pred = inp
        # print(f"Writing results for pair: {pair}")
        with h5py.File(str(match_path), "a", libver="latest") as fd:
            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)
            matches = pred["matches0"].cpu().short().numpy()
            grp.create_dataset("matches0", data=matches)
            if "matching_scores0" in pred:
                scores = pred["matching_scores0"].cpu().half().numpy()
                grp.create_dataset("matching_scores0", data=scores)
    except Exception as e:
        print(f"Error in writer_fn: {str(e)}")
        import traceback
        traceback.print_exc()

def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Matching local features with configuration:\n{}".format(pprint.pformat(conf))
    )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q} not found.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref} not found.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching as everything is already computed.")
        return match_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=10000, trust_repo='check').to(device)
    xfeat = XFeat()
    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {k: v.to(device, non_blocking=True) if not k.startswith("image") else v for k, v in data.items()}
        name0, name1 = pairs[idx]
        pair = names_to_pair(name0, name1)
        overall_keypoints0 = data["keypoints0"].squeeze().cpu().numpy()
        overall_keypoints1 = data["keypoints1"].squeeze().cpu().numpy()

        image_size0 = data.get("image_size0")
        image_size1 = data.get("image_size1")
        equirect_coords1 = []
        equirect_coords2 = []
        out0 = []
        out1 = []
        face1_list = []
        face2_list = []
        t_intial = time.time()
        for face1 in faces_order:
            for face2 in faces_order:
                face1_list.append(face1)
                face2_list.append(face2)
                img0_face_keypoints = data.get(f"keypoints_{face1}0", torch.empty(0, 2)).to(device).squeeze()
                img1_face_keypoints = data.get(f"keypoints_{face2}1", torch.empty(0, 2)).to(device).squeeze()
                img0_face_descriptors = data.get(f"descriptors_{face1}0", torch.empty(0, 2)).to(device).squeeze()
                img1_face_descriptors = data.get(f"descriptors_{face2}1", torch.empty(0, 2)).to(device).squeeze()
                img0_face_scores = data.get(f"scores_{face1}0", torch.empty(0, 2)).to(device).squeeze()
                img1_face_scores = data.get(f"scores_{face2}1", torch.empty(0, 2)).to(device).squeeze()
                output0 = {
                                'keypoints': img0_face_keypoints,
                                'descriptors': img0_face_descriptors,
                                'scores': img0_face_scores,
                                'image_size': (image_size0[0][0]/4,image_size0[0][1]/2)
                            }
                output1 = {
                    'keypoints': img1_face_keypoints,
                    'descriptors': img1_face_descriptors,
                    'scores': img1_face_scores,
                    'image_size': (image_size0[0][0]/4,image_size0[0][1]/2)
                }
                out0.append(output0)
                out1.append(output1)

        t_inputs = time.time()
        # print (f"Time for inputs {t_inputs-t_intial}")
        res = xfeat.batch_match_lighterglue(out0, out1)
        t_model = time.time()
        # print (f"Time for Model {t_model - t_inputs}")

        t1 = time.time()
        for i, r in enumerate(res):
            mkpts0 = r["mkpts_0"]  # shape (M, 2)
            mkpts1 = r["mkpts_1"]  # shape (M, 2)

            face1 = face1_list[i]
            face2 = face2_list[i]

            # Separate x and y for batch processing
            x1, y1 = mkpts0[:, 0], mkpts0[:, 1]
            x2, y2 = mkpts1[:, 0], mkpts1[:, 1]

            # Convert all keypoints in a batch
            equirect_coords1_batch = cubemap_to_equirectangular_uv_batch(
                face1.upper(), x1, y1, image_size0[0][0] / 4, image_size0[0][0], image_size0[0][1]
            )
            equirect_coords2_batch = cubemap_to_equirectangular_uv_batch(
                face2.upper(), x2, y2, image_size0[0][0] / 4, image_size0[0][0], image_size0[0][1]
            )

            # Append to the lists
            equirect_coords1.extend(equirect_coords1_batch)
            equirect_coords2.extend(equirect_coords2_batch)
        t2 = time.time()
        # print ("Loop Time: ",t2-t1)
        params = [0, image_size0.cpu().numpy()[0][0]/2, image_size0.cpu().numpy()[0][1]/2]
        try:
            points0_spherical = cam_from_img_vectorized(params, np.array(equirect_coords1))
            points1_spherical = cam_from_img_vectorized(params, np.array(equirect_coords2))
            t1 = time.time()
            inliers, num_inliers = ransac.get_inliers(points0_spherical.T, points1_spherical.T)
            # t2 = time.time()
            # print ("Ransac Time",t2-t1)
        except Exception as e:
            print (len(equirect_coords1))
            print (e)
            ransac.reset()
                
            num_keypoints = overall_keypoints0.shape[0] 
            matches0 = -np.ones(num_keypoints, dtype=np.int64)
            matching_scores0 = np.zeros(num_keypoints, dtype=np.float32)

            pred = {
                'matches0': torch.from_numpy(matches0),
                'matching_scores0': torch.from_numpy(matching_scores0),
            }
            writer_queue.put((pair, pred))

            continue
        ransac.reset()
        # print (points1_spherical.shape,np.sum(inliers))
        # Filter matches to inliers
        equirect_coords1 = np.array(equirect_coords1)[inliers]
        equirect_coords2 = np.array(equirect_coords2)[inliers]
        # print (overall_keypoints0.shape)
        num_keypoints = overall_keypoints0.shape[0]
        matches0 = -np.ones(num_keypoints, dtype=np.int64)
        matching_scores0 = np.zeros(num_keypoints, dtype=np.float32)
        # print (matches0.shape)
        if len(equirect_coords1) > 0:
            # The matches are already paired in equirect_coords1 and equirect_coords2
            # Just need to find the corresponding keypoint indices
            # print ()
            t1 =time.time()
            # for i, (coord1, coord2) in enumerate(zip(equirect_coords1, equirect_coords2)):
            #     # Find corresponding keypoints
            #     kp0_idx = np.argmin(np.sum((overall_keypoints0 - coord1) ** 2, axis=1))
            #     kp1_idx = np.argmin(np.sum((overall_keypoints1 - coord2) ** 2, axis=1))
            #     matches0[kp0_idx] = kp1_idx
            overall_keypoints0 = np.asarray(overall_keypoints0, dtype='float32')
            overall_keypoints1 = np.asarray(overall_keypoints1, dtype='float32')
            equirect_coords1 = np.asarray(equirect_coords1, dtype='float32')
            equirect_coords2 = np.asarray(equirect_coords2, dtype='float32')

            tree0 = cKDTree(overall_keypoints0)
            tree1 = cKDTree(overall_keypoints1)

            # Query nearest neighbors
            _, indices0 = tree0.query(equirect_coords1, k=1)  # k=1 for nearest neighbor
            _, indices1 = tree1.query(equirect_coords2, k=1)

            # Update matches
            matches0[indices0] = indices1
            t2 = time.time()
            # print ("Nearest Neighbour time:",t2-t1)
        pred = {
            'matches0': torch.from_numpy(matches0),
            'matching_scores0': torch.from_numpy(matching_scores0),
        }
        writer_queue.put((pair, pred))

    writer_queue.join()
    logger.info("Finished exporting matches.")
    return match_path


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                f"Provide an export_dir if features is not a file path: {features}"
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="output_features")
    parser.add_argument("--matches", type=Path)
    conf = {"output": "matches"}
    args = parser.parse_args()
    main(conf, args.pairs, args.features, args.export_dir, args.matches)
