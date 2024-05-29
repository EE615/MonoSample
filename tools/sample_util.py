import numpy as np
import open3d as o3d
import pathlib
import re
import random
import cv2
import pandas as pd
import pickle
from lib.datasets.kitti_utils import Calibration
from copy import deepcopy
from tools.dataset_util import Dataset
from sklearn.decomposition import PCA
from tools.box_util import boxes_bev_iou_cpu, rect2lidar, check_points_in_boxes3d, bbox3d_to_corners_3d
from lib.datasets.kitti_utils import Object3d


def merge_labels(labels, samples, calib_, image_shape):
    labels += [sample.to_label() for sample in samples]
    occlusion =  recompute_occlusion(labels, calib_, image_shape)
    for i, label in enumerate(labels):
        label.occlusion = area2occlusion(occlusion[i])
        label.level = label.get_obj_level()
    return labels

def recompute_occlusion(labels, calib_, image_shape=np.array([375, 1242, 3])):
    canvas = np.zeros(image_shape[:2], dtype=np.int8) - 1
    labels = sorted(labels, key=lambda x: x.pos[2], reverse=True)
    area = []
    for i, label in enumerate(labels):
        corners = label.generate_corners3d()
        uv, _ = calib_.rect_to_img(corners)
        u_min = round(max(0, np.min(uv[:, 0])))
        v_min = round(max(0, np.min(uv[:, 1])))
        u_max = round(min(np.max(uv[:, 0]), image_shape[1]))
        v_max = round(min(np.max(uv[:, 1]), image_shape[0]))

        canvas[v_min: v_max, u_min: u_max] = i
        area.append( (v_max - v_min) * (u_max - u_min) + 1e-6)
    for i, label in enumerate(labels):
        current_area = np.sum(canvas == i)
        area[i] = 1 - current_area / area[i]
        if area[i] < 0 or area[i] > 1:
            area[i] = 1.0
    return area

def area2occlusion(area):
    if area < 0.1:
        return 0
    elif area < 0.4:
        return 1
    elif area < 0.8:
        return 2
    else:
        return 3


def to3d(image, depth, calib, bbox2d=None):
    assert image.shape[:2] == depth.shape
    h, w = depth.shape
    u = np.repeat(np.arange(w), h)
    v = np.tile(np.arange(h), w)
    d = depth[v, u]
    rgb = image[v, u][:, ::-1]
    if bbox2d:
        u += bbox2d[0]
        v += bbox2d[1]
    cord = calib.img_to_rect(u, v, d)
    return cord, rgb


def to2d(cord, rgb, calib, image_shape):
    assert cord.shape[0] == rgb.shape[0]
    assert cord.shape[1] == 3, rgb.shape[1] == 3
    h, w = image_shape[:2]
    uv, d = calib.rect_to_img(cord)
    uv = np.round(uv).astype(int)
    valid = (0 <= uv[:, 1]) & (uv[:, 1] < h) & (0 <= uv[:, 0]) & (uv[:, 0] < w)
    u, v = uv[valid].T
    rgb = rgb[valid]
    d = d[valid]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), np.inf)

    for i in range(len(u)):
        if d[i] < depth[v[i], u[i]]:
            depth[v[i], u[i]] = d[i]
            image[v[i], u[i]] = rgb[i]

    depth[depth == np.inf] = 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [largest_contour], color=255)
    u_min, v_min, w_, h_ = cv2.boundingRect(largest_contour)
    v_max = v_min + h_
    u_max = u_min + w_
    image = image[v_min: v_max, u_min: u_max]
    depth = depth[v_min: v_max, u_min: u_max]
    mask = mask[v_min: v_max, u_min: u_max]
    thresh = thresh[v_min: v_max, u_min: u_max]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=1) 
    eroded = cv2.erode(dilated, kernel, iterations=1)
    depth = (depth * 256).astype(np.uint16)
    mask = ((eroded == 255) & (thresh == 0)).astype(np.uint8)
    # image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    # depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
    depth = depth.astype(np.float32) / 256.0

    return image, depth, [u_min, v_min, u_max, v_max]


class SampleDatabase:
    def __init__(self,
                 database_path,
                 idx_list=None,
                 config=None,
                 ):
        self.database_path = pathlib.Path(database_path)
        assert self.database_path.exists()
        self.image_path = self.database_path / "image"
        self.depth_path = self.database_path / "depth"
        # self.mask_path = self.database_path / "mask"
        with open(self.database_path / "kitti_car_database_with_flip.pkl", "rb") as f:
            database = pd.read_pickle(f)
        with open(self.database_path / "sample_image_database_with_flip.pkl", "rb") as f:
            self.sample_image_database = pickle.load(f)
        with open(self.database_path / "sample_depth_dense_database_with_flip.pkl", "rb") as f:
            self.sample_depth_database = pickle.load(f)

        default_config = {
            "prob": 0.5,
            "database_num": -1,
            'sample_num': 10,
            'sample_constraint': {
                "max_z2y": 10,
                'max_x2z': 10,
                'max_dz': 10,
                'max_rate': 1.2,
                'min_rate': 0.5,
            },
            'position_sample_num': 40,
        }
        self.config = {**default_config, **(config if config is not None else {})}
        if idx_list is not None:
            database = database[database['idx'].isin(idx_list)]
        database_num = self.config["database_num"]
        self.database = database.sample(n=database_num) if database_num != -1 else database
        self.pointer = len(database)
        self.indices = None
        self.position_sample_num = self.config["position_sample_num"]
        self.sample_num = self.config["sample_num"]


    @staticmethod
    def get_ry_(alpha, xyz_, calib_):
        uv_, _ = calib_.rect_to_img(xyz_.reshape(1, -1))
        ry_ = calib_.alpha2ry(alpha, uv_[:, 0])
        return ry_

    @staticmethod
    def get_y_on_plane(x, z, plane):
        a, b, c, d = plane
        y = - a * x - c * z - d
        y /= b
        return y

    def samples_with_range(self, xyz_, max_z2y=2, max_x2z=10, max_dz=10, max_rate=2, min_rate=0.5):
        n = xyz_.shape[0]
        xyz_ = xyz_[np.newaxis, :]  # (1, n, 3)
        assert n <= self.position_sample_num
        df_ = self.database.sample(frac=0.1)

        def func(x):
            x = df_[x].to_numpy()
            x = np.repeat(x[:, np.newaxis], n, axis=1)
            return x

        z2y = func('z/y')
        x2z = func('x/z')
        z = func('z')
        h = func('h')

        x2z_ = np.arctan2(xyz_[:, :, 0], xyz_[:, :, 2]) * 180 / np.pi # (1, n)
        z_ = xyz_[:, :, 2]

        z2y_ = np.arctan2(xyz_[:, :, 2], xyz_[:, :, 1] - h / 2) * 180 / np.pi
        condition = ((np.abs(z2y - z2y_) < max_z2y) &
                     (np.abs(x2z - x2z_) < max_x2z) &
                     (np.abs(z - z_) < max_dz) &
                     (z / z_ < max_rate) & (z / z_ > min_rate))

        samples = []
        xyz_list = []
        indexes = [np.where(condition[:, col])[0] for col in range(n)]
        xyz_ = xyz_[0]  # (n, 3)
        for i, index in enumerate(indexes):
            if index.shape[0] == 0:
                continue
            sample = df_.iloc[np.random.choice(index)]
            samples.append(sample)
            xyz_list.append(xyz_[i])

        return samples, np.array(xyz_list)

    def samples_from_database(self, num):
        pointer, indices, database = self.pointer, self.indices, self.database
        if pointer >= len(database):
            indices = np.random.permutation(len(database))
            pointer = 0
        samples = [database.iloc[idx] for idx in indices[pointer: pointer + num]]
        if len(samples) < num:
            samples += [database.iloc[idx] for idx in indices[: num - len(samples)]]

        pointer += len(samples)
        self.pointer = pointer
        self.indices = indices
        return samples

    def xyz_to_bbox3d(self, samples, xyz_, calib_):
        sample_num = len(samples)
        if sample_num == 0:
            return [], np.zeros((0, 7))

        calib = np.array([s['calib'] for s in samples])
        ry = np.array([[s['label'].ry] for s in samples])
        xyz = np.array([s['label'].pos for s in samples])
        bbox2d = np.array([s['bbox2d'] for s in samples])
        alpha = np.array([[s['label'].alpha] for s in samples])
        lhw = np.array([[s['label'].l, s['label'].h, s['label'].w] for s in samples])

        ry_ = np.array([self.get_ry_(alpha[i], xyz_[i], calib_) for i in range(sample_num)])
        bbox3d_ = np.concatenate([xyz_, lhw, ry_], axis=1)

        return samples, bbox3d_

    def sample_xyz(self, plane_=None, samples=None, xyz_=None):
        if samples is not None and xyz_ is not None:
            assert len(samples) == xyz_.shape[0]
        sample_num = self.position_sample_num if samples is None else len(samples)
        sample_num = sample_num if xyz_ is None else xyz_.shape[0]
        if samples is None:
            samples = self.samples_from_database(sample_num)
        if xyz_ is None:
            assert plane_ is not None
            low_z, high_z = self.z_range
            low_x, high_x = self.x_range
            x_ = np.random.uniform(low=low_x, high=high_x, size=(sample_num, 1))
            z_ = np.random.uniform(low=low_z, high=high_z, size=(sample_num, 1))
            y_ = np.array([self.get_y_on_plane(x_[i], z_[i], plane_) for i in range(sample_num)])
            xyz_ = np.concatenate([x_, y_, z_], axis=1)
        return samples, xyz_

    @staticmethod
    def get_scene_type(pos):
        z = pos[:, 1]
        segments = {
            "a": (70 > z) & (z >= 45),  # 3049
            "b": (45 > z) & (z >= 30),  # 1846
            "c": (30 > z) & (z >= 15),  # 2057
            "d": (15 > z) & (z >= 0)  # 529
        }
        grid_sums = {key: np.sum(value) for key, value in segments.items()}
        scene_type = max(grid_sums, key=grid_sums.get)
        return scene_type

    def get_valid_grid(self, grid):
        pos2d = np.array(list(grid.keys()))
        dis = np.linalg.norm(pos2d, axis=1)

        scene_type = self.get_scene_type(pos2d)

        valid = dis < min(np.max(dis) - 10, 60)  # Points near the maximum distance are unreliable
        pos2d = pos2d[valid]

        state = {
            'a': lambda x: (x[:, 1] >= 5) & (x[:, 0] >= -20) & (x[:, 0] <= 20),
            'b': lambda x: (x[:, 1] >= 5) & (x[:, 0] >= -15) & (x[:, 0] <= 15),
            'c': lambda x: (x[:, 1] >= 5) & (x[:, 0] >= -10) & (x[:, 0] <= 10),
            'd': lambda x: (x[:, 1] >= 5) & (x[:, 0] >= -10) & (x[:, 0] <= 10)
        }
        valid = state[scene_type](pos2d)
        pos2d = pos2d[valid]

        return pos2d, scene_type

    def sample_from_grid(self, grid, grid_size=1., max_sample_num=40):
        pos2d, scene_type = self.get_valid_grid(grid)
        grid_sum = pos2d.shape[0]

        sample_num = min(grid_sum // 10, max_sample_num)

        indices = np.random.choice(pos2d.shape[0], sample_num, replace=False)
        offset = np.random.uniform(-grid_size / 2, grid_size / 2, size=(sample_num, 2))
        pos2d = pos2d[indices]

        plane_ = [grid[(pos2d[i][0], pos2d[i][1])]["plane"] for i in range(sample_num)]
        x_, z_ = (pos2d + offset).T
        y_ = np.array([self.get_y_on_plane(x_[i], z_[i], plane_[i]) for i in range(sample_num)])

        xyz_ = np.vstack((x_, y_, z_)).T

        samples, xyz_ = self.samples_with_range(xyz_, **self.config["sample_constraint"])
        return samples, xyz_, scene_type

    @staticmethod
    def check_normal_angle(normal, max_degree):
        assert normal.shape[0] == 3
        limit = np.cos(np.radians(max_degree))
        norm = np.linalg.norm(normal)
        cos = np.abs(normal[1]) / norm
        return cos >= limit

    @staticmethod
    def sample_put_on_plane(bbox3d, ground, radius=3, min_num=25, max_var=0.5e-2, max_degree=20):
        bbox3d = bbox3d.copy()
        flag = np.zeros((bbox3d.shape[0]), dtype=bool)
        for i, pos in enumerate(bbox3d[:, :3]):
            distance = np.linalg.norm(ground - pos, axis=1)
            nearby = ground[distance < radius]
            if nearby.shape[0] < min_num:
                continue

            pca = PCA(n_components=3)
            pca.fit(nearby)
            normal = pca.components_[2]
            var = pca.explained_variance_ratio_[2]
            if var > max_var:
                continue
            if not SampleDatabase.check_normal_angle(normal, max_degree):
                continue
            d = -normal.dot(np.mean(nearby, axis=0))
            bbox3d[i, 1] = SampleDatabase.get_y_on_plane(pos[0], pos[2], [*normal, d])
            flag[i] = True
        return bbox3d, flag

    def get_samples(self, ground, non_ground, calib_, plane_, grid=None, ues_plane_filter=True, origin_label=None):
        if grid is None:
            samples, xyz_ = self.sample_xyz(plane_)
            ues_plane_filter = True
        else:
            samples, xyz_, scene_type = self.sample_from_grid(grid, max_sample_num=self.position_sample_num)

        samples, bbox3d_ = self.xyz_to_bbox3d(samples, xyz_, calib_)

        flag1 = np.ones((bbox3d_.shape[0]), dtype=bool)

        if ues_plane_filter:
            bbox3d_, flag1 = self.sample_put_on_plane(bbox3d_, ground, radius=3, min_num=10, max_degree=15)

        if flag1.sum() == 0:
            return []

        #  bbox3d should be in lidar coordinate system
        bbox3d_in_lidar = rect2lidar(bbox3d_[flag1], calib_)

        iou = boxes_bev_iou_cpu(bbox3d_in_lidar, bbox3d_in_lidar)
        iou[range(bbox3d_in_lidar.shape[0]), range(bbox3d_in_lidar.shape[0])] = 0
        rows, cols = np.triu_indices(n=iou.shape[0], k=1)
        iou[rows, cols] = 0
        flag2 = iou.max(axis=1) == 0
        if flag2.sum() == 0:
            return []

        points_in_lidar = calib_.rect_to_lidar(non_ground)
        flag3 = ~ check_points_in_boxes3d(points_in_lidar, bbox3d_in_lidar[flag2])
        if flag3.sum() == 0:
            return []

        valid = np.arange(bbox3d_.shape[0])[flag1][flag2][flag3]
        valid = np.random.choice(valid, min(self.sample_num, len(valid)), replace=False)
        res = [Sample(samples[i], bbox3d_[i], calib_, self) for i in valid]

        return res

    @staticmethod
    def get_merged_points(samples, image, depth, calib):
        cords, rgbs = [], []
        for sample in samples:
            cord, rgb = sample.get_points()
            cords.append(cord)
            rgbs.append(rgb)
        cord, rgb = to3d(image, depth, calib)
        cord = np.concatenate([cord, *cords], axis=0)
        rgb = np.concatenate([rgb, *rgbs], axis=0)
        return cord, rgb

    @staticmethod
    def add_samples_to_scene(samples, image, depth, use_edge_blur=False):
        image_, depth_ = image.copy(), depth.copy()
        samples = sorted(samples, key=lambda x: x.bbox3d_[2], reverse=False)  # z 降序
        mask = np.zeros(image.shape[:2], dtype=bool)
        flag = np.zeros(len(samples), dtype=bool)
        for i, sample in enumerate(samples):
            image_, depth_, mask, flag[i] = sample.cover(image_, depth_, mask)

            if use_edge_blur:
                blur = cv2.GaussianBlur(image_, (7, 7), 0)
                kernel = np.ones((7, 7), np.uint8)
                mask_ = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                blur_place = mask_.astype(bool) != mask
                image_[blur_place] = blur[blur_place]

        return image_, depth_, [sample for i, sample in enumerate(samples) if flag[i]]


class Sample:
    def __init__(self, sample, bbox3d, calib, database: SampleDatabase):
        # suffix _:  value from augmented scene
        # no suffix: value from sample's scene
        self.sample = sample
        self.bbox3d_ = bbox3d
        self.database = database

        self.label = deepcopy(sample['label'])
        self.calib = deepcopy(sample['calib'])
        self.alpha_ = self.label.alpha
        self.calib_ = deepcopy(calib)
        self.plane = deepcopy(sample['plane'])
        self.bbox2d = deepcopy(sample['bbox2d'])
        self.name = sample['name']
        self.image_shape = deepcopy(sample['image_shape'])

        self.flipped = sample['flipped']
        self.image = self.get_image()
        self.depth = self.get_depth()

        self.occlusion_ = 0
        self.trucation_ = 0
        self.image_, self.depth_, self.bbox2d_ = self.transform()

    def __repr__(self):
        return f"Sample(name={self.name})"

    def get_image(self):
        try:
            image = self.database.sample_image_database[self.name]
        except KeyError:
            image_file = self.database.image_path / (self.name + ".png")
            assert image_file.exists()
            image = cv2.imread(str(image_file))
        return image

    def get_depth(self):
        try:
            depth = self.database.sample_depth_database[self.name]
        except KeyError:
            depth_file = self.database.depth_path / (self.name + ".png")
            assert depth_file.exists()
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 256.0
        return depth

    def get_points(self, use_transform=False, use_source=False):
        if not use_transform:
            assert self.depth.shape[:2] == self.image.shape[:2]
            image, depth, calib, label = self.image, self.depth, self.calib, self.label
            calib_, bbox3d_, bbox2d = self.calib_, self.bbox3d_, self.bbox2d
            xyz, ry = label.pos, label.ry
            xyz_, ry_ = bbox3d_[:3], bbox3d_[6]

            cord, rgb = to3d(image, depth, calib, bbox2d)

            valid = cord[:, 2] >= 1e-3
            cord, rgb = cord[valid], rgb[valid]
            if use_source:
                return cord, rgb

            cord = cord - xyz
            dry = ry_ - ry
            Ry = np.array([[np.cos(dry), 0, np.sin(dry)],
                           [0, 1, 0],
                           [-np.sin(dry), 0, np.cos(dry)]])

            cord = cord @ Ry.T + xyz_
        else:
            assert self.depth_.shape[:2] == self.image_.shape[:2]
            image_, depth_, calib, label = self.image_, self.depth_, self.calib, self.label
            calib_, bbox3d_, bbox2d, bbox2d_ = self.calib_, self.bbox3d_, self.bbox2d, self.bbox2d_
            cord, rgb = to3d(image_, depth_, calib_, bbox2d_)
            valid = cord[:, 2] >= 1e-3
            cord, rgb = cord[valid], rgb[valid]

        return cord, rgb

    @staticmethod
    def get_3d_center_in_2d(xyz, calib):
        xyz = xyz.reshape(1, -1)[:, :3]
        uv, _ = calib.rect_to_img(xyz)
        uv = uv.reshape(2)
        return uv

    def transform_in_3d(self):
        assert self.depth.shape[:2] == self.image.shape[:2]
        cord, rgb = self.get_points(use_source=True)

        calib, label = self.calib, self.label
        calib_, bbox3d_, bbox2d = self.calib_, self.bbox3d_, self.bbox2d

        xyz, ry = label.pos, label.ry
        xyz_, ry_ = bbox3d_[:3], bbox3d_[6]

        uv_tmp, _ = calib.rect_to_img((xyz_ - [0, label.h / 2, 0]).reshape(1, -1))
        u_tmp = uv_tmp[0, 0:1]
        uv_tmp, _ = calib.rect_to_img((xyz - [0, label.h / 2, 0]).reshape(1, -1))
        v_tmp = uv_tmp[0, 1:2]
        xyz_tmp = calib.img_to_rect(u_tmp, v_tmp, xyz[2:])
        xyz_tmp = xyz_tmp.reshape(3) + [0, label.h / 2, 0]
        u_tmp = u_tmp[0]
        v_tmp = v_tmp[0]

        dry = ry_ - ry
        rx = np.arctan2(xyz[2], xyz[1] - label.h / 2)
        rx_ = np.arctan2(xyz_[2], xyz_[1] - label.h / 2)
        drx = - (rx_ - rx)
        print(drx * 180 / np.pi)

        Ry = np.array([[np.cos(dry), 0, np.sin(dry)],
                       [0, 1, 0],
                       [-np.sin(dry), 0, np.cos(dry)]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(drx), -np.sin(drx)],
                       [0, np.sin(drx), np.cos(drx)]])

        cord = (cord - xyz + [0, label.h / 2, 0]) @ Ry.T @ Rx.T + xyz_tmp - [0, label.h / 2, 0]

        image, depth, bbox2d_tmp = to2d(cord, rgb, calib, self.image_shape)

        center_tmp = self.get_3d_center_in_2d(xyz_tmp + [0, -label.h / 2, 0], calib)
        center_ = self.get_3d_center_in_2d(bbox3d_[:3] + [0, -label.h / 2, 0], calib_)
        depth_ = depth - xyz_tmp[2] + xyz_[2]
        depth_[depth < 1e-2] = 0
        h, w = image.shape[:2]
        rate = (xyz_[2] / calib_.fv) / (xyz_tmp[2] / calib.fv)
        h_, w_ = round(h / rate), round(w / rate)
        image_ = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_NEAREST)
        depth_ = cv2.resize(depth_, (w_, h_), interpolation=cv2.INTER_NEAREST)

        bbox2d_ = np.tile((bbox2d_tmp[:2] - center_tmp) / rate + center_, 2)
        bbox2d_ = np.round(bbox2d_).astype(int)
        bbox2d_[2:] += [w_, h_]

        return image_, depth_, bbox2d_.tolist()

    def transform(self):
        assert self.depth.shape[:2] == self.image.shape[:2]
        image, depth, calib, label = self.image, self.depth, self.calib, self.label
        calib_, bbox3d_, bbox2d, alpha_ = self.calib_, self.bbox3d_, self.bbox2d, self.alpha_

        center = self.get_3d_center_in_2d(label.pos + [0, -label.h / 2, 0], calib)
        center_ = self.get_3d_center_in_2d(bbox3d_[:3] + [0, -label.h / 2, 0], calib_)

        dry = bbox3d_[6] - label.ry  # ry_ - ry
        h, w = depth.shape

        offset = np.arange(w, dtype=int) - center[0] + bbox2d[0]

        width = abs(np.sin(alpha_) * label.w) + abs(np.cos(alpha_) * label.l)
        offset = - np.tan(dry) * offset * width / w

        depth_ = depth - label.pos[2] + bbox3d_[2] + offset.reshape(1, -1)
        depth_[depth < 1e-2] = 0

        rate = (bbox3d_[2] / calib_.fv) / (label.pos[2] / calib.fv)  # z_ / z
        h_, w_ = round(h / rate), round(w / rate)

        depth_ = cv2.resize(depth_, (w_, h_), interpolation=cv2.INTER_NEAREST)
        image_ = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_NEAREST)

        bbox2d_ = np.tile((bbox2d[:2] - center) / rate + center_, 2)
        bbox2d_ = np.round(bbox2d_).astype(int)
        bbox2d_[2:] += [w_, h_]

        return image_, depth_, bbox2d_.tolist()

    @staticmethod
    def truncate(image_, depth_, bbox2d_, image_shape):
        h, w = image_shape[:2]
        u_min, v_min, u_max, v_max = bbox2d_
        area = (v_max - v_min) * (u_max - u_min)
        if u_min < 0:
            image_ = image_[:, -u_min:]
            depth_ = depth_[:, -u_min:]
            bbox2d_[0] = 0
        if v_min < 0:
            image_ = image_[-v_min:, :]
            depth_ = depth_[-v_min:, :]
            bbox2d_[1] = 0
        if u_max > w:
            image_ = image_[:, :w - u_max]
            depth_ = depth_[:, :w - u_max]
            bbox2d_[2] = w
        if v_max > h:
            image_ = image_[:h - v_max, :]
            depth_ = depth_[:h - v_max, :]
            bbox2d_[3] = h
        area_ = (bbox2d_[3] - bbox2d_[1]) * (bbox2d_[2] - bbox2d_[0])
        truncate_rate = (area - area_) / area
        return image_, depth_, bbox2d_, truncate_rate

    def cover(self, image, depth, mask, area_threshold=0.5):
        assert image.shape[:2] == depth.shape
        blank_rgb, blank_d, mask = image.copy(), depth.copy(), mask.copy()
        image_, depth_, bbox2d_ = self.image_, self.depth_, self.bbox2d_
        image_, depth_, bbox2d_, self.trucation_ = self.truncate(image_, depth_, bbox2d_, image.shape)

        u_min, v_min, u_max, v_max = bbox2d_
        if u_min >= u_max or v_min >= v_max:
            return blank_rgb, blank_d, mask, False

        d_in_bbox2d = blank_d[v_min: v_max, u_min: u_max]
        valid = (depth_ > 1e-2) & (depth_ < d_in_bbox2d)
        area = (v_max - v_min) * (u_max - u_min) - np.sum(depth_ <= 1e-2)
        valid_rate = np.sum(valid) / area
        if valid_rate <= area_threshold:
            return blank_rgb, blank_d, mask, False

        blank_rgb[v_min: v_max, u_min: u_max][valid] = image_[valid]
        blank_d[v_min: v_max, u_min: u_max][valid] = depth_[valid]
        mask[v_min: v_max, u_min: u_max][valid] = True

        return blank_rgb, blank_d, mask, True

    def to_label(self):
        label = self.label
        cls = label.cls_type
        trucation = self.trucation_
        score = 0
        occlusion = 0
        x_, y_, z_, l_, h_, w_, ry_ = self.bbox3d_
        alpha = self.get_alpha(self.bbox3d_[:3], ry_, self.calib_)
        u_min, v_min, u_max, v_max = self.bbox2d_
        line = f"{cls} {trucation} {occlusion} {alpha} {u_min} {v_min} {u_max} {v_max} {h_} {w_} {l_} {x_} {y_} {z_} {ry_} {score}"
        res = Object3d(line)
        res.is_fake = True
        return res

    @staticmethod
    def get_alpha(xyz, ry, calib):
        uv, _ = calib.rect_to_img(xyz.reshape(1, -1))
        alpha = calib.ry2alpha(ry, uv[:, 0])[0]
        return alpha


from pathlib import Path
import time
import datetime

if __name__ == '__main__':
    test_dir = Path("/mnt/e/DataSet/kitti/kitti_drx_database/test")
    np.random.seed(2)

    database = SampleDatabase("/mnt/e/DataSet/kitti/kitti_drx_database/")
    dataset = Dataset("train", r"/mnt/e/DataSet/kitti")
    mean_samples = 0
    n = 200
    dt = 0
    for idx in range(n):
        calib_ = dataset.get_calib(idx)
        image, depth = dataset.get_image_with_depth(idx, use_penet=False)
        ground, non_ground = dataset.get_lidar_with_ground(idx, fov=True)
        plane_ = dataset.get_plane(idx)
        grid = dataset.get_grid(idx)
        _, _, labels = dataset.get_bbox(idx, chosen_cls=["Car", 'Van', 'Truck', 'DontCare'])

        time1 = time.time()
        samples = database.get_samples(ground, non_ground, calib_, plane_, grid=grid, origin_label=labels)
        image_, depth_, samples = database.add_samples_to_scene(samples, image, depth, use_edge_blur=True)
        # image_, depth_, samples = database.add_samples_to_scene(samples, image, depth, calib=calib_, use_3d_projection=True)
        labels = merge_labels(labels, samples, calib_, image.shape)
        time2 = time.time()

        # for label in labels:
        #     cv2.putText(image_, str(round(label.pos[-1], 2)), (int(label.box2d[0]), int(label.box2d[1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        mean_samples += len(samples)
        cv2.imwrite(str(test_dir / ('%06d.png' % idx)), image_)
        dt += time2 - time1
    print("time: ", dt / n)
    print(mean_samples / n)
