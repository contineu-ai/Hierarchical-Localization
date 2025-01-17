import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from cupyx.scipy import ndimage as cupyx_ndimage


# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU-based converter class (as provided earlier)
class GPU_Convert:
    def __init__(self, image_shape):
        h, w, _ = image_shape
        SQUARE_SIDE = h // 2
        self.cp = __import__('cupy')
        self.cupyx_ndimage = cupyx_ndimage
        self.coors_xy = self.uv2coor(self.xyz2uv(self.xyzcube(SQUARE_SIDE)), h, w)
        # logger.info("Running inference on GPU")

    def xyzcube(self, face_w):
        out = self.cp.zeros((face_w, face_w * 6, 3), dtype=self.cp.float32)
        rng = self.cp.linspace(-0.5, 0.5, num=face_w, dtype=self.cp.float32)
        grid = self.cp.stack(self.cp.meshgrid(rng, -rng), -1)

        # Faces of the cube
        out[:, 0 * face_w:1 * face_w, [0, 1]] = grid  # Front
        out[:, 0 * face_w:1 * face_w, 2] = 0.5
        grid_r = self.cp.flip(grid, axis=1)
        out[:, 1 * face_w:2 * face_w, [2, 1]] = grid_r  # Right
        out[:, 1 * face_w:2 * face_w, 0] = 0.5
        grid_b = self.cp.flip(grid, axis=1)
        out[:, 2 * face_w:3 * face_w, [0, 1]] = grid_b  # Back
        out[:, 2 * face_w:3 * face_w, 2] = -0.5
        out[:, 3 * face_w:4 * face_w, [2, 1]] = grid  # Left
        out[:, 3 * face_w:4 * face_w, 0] = -0.5
        grid_u = self.cp.flip(grid, axis=0)
        out[:, 4 * face_w:5 * face_w, [0, 2]] = grid_u  # Up
        out[:, 4 * face_w:5 * face_w, 1] = 0.5
        out[:, 5 * face_w:6 * face_w, [0, 2]] = grid  # Down
        out[:, 5 * face_w:6 * face_w, 1] = -0.5

        return out

    def xyz2uv(self, xyz):
        x, y, z = self.cp.split(xyz, 3, axis=-1)
        u = self.cp.arctan2(x, z)
        c = self.cp.sqrt(x**2 + z**2)
        v = self.cp.arctan2(y, c)
        return self.cp.concatenate([u, v], axis=-1)

    def uv2coor(self, uv, h, w):
        u, v = self.cp.split(uv, 2, axis=-1)
        coor_x = (u / (2 * self.cp.pi) + 0.5) * w - 0.5
        coor_y = (-v / self.cp.pi + 0.5) * h - 0.5
        return self.cp.concatenate([coor_x, coor_y], axis=-1)

    def sample_equirec(self, e_img, coor_xy, order):
        w = e_img.shape[1]
        coor_x, coor_y = self.cp.split(coor_xy, 2, axis=-1)
        pad_u = self.cp.roll(e_img[[0]], w // 2, axis=1)
        pad_d = self.cp.roll(e_img[[-1]], w // 2, axis=1)
        e_img = self.cp.concatenate([e_img, pad_d, pad_u], axis=0)
        result = self.cupyx_ndimage.map_coordinates(
            e_img, self.cp.array([coor_y, coor_x]), order=order, mode='wrap'
        )[..., 0]
        return result

    def e2c(self, e_img, coor_xy):
        c = e_img.shape[2]
        e_img = self.cp.asarray(e_img)
        coor_xy = self.cp.asarray(coor_xy)
        cubemap = self.cp.stack(
            [self.sample_equirec(e_img[..., i], coor_xy, order=1) for i in range(c)],
            axis=-1,
        )
        cubemap_faces = self.cp.array_split(cubemap, 6, axis=1)
        cubemap_dict = {
            k: self.cp.asnumpy(cubemap_faces[i])
            for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
        }
        return cubemap_dict

    def convert_to_cubemaps(self, img):
        return self.e2c(img, self.coors_xy)


# Convert cubemap to equirectangular
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