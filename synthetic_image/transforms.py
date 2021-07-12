import cv2
import math
import numpy as np
from PIL import Image

from imgaug.parameters import StochasticParameter, handle_continuous_param
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmenters import Augmenter

class Interpolation:
    NEAREST = 0
    LINEAR = 1
    BICUBIC = 3

def _inter2pil(inter):
    if inter == Interpolation.BICUBIC:
        return Image.BICUBIC
    elif inter == Interpolation.LINEAR:
        return Image.LINEAR
    elif inter == Interpolation.NEAREST:
        return Image.NEAREST
    else:
        raise ValueError('Invalid interpolation type.')

def _inter2cv2(inter):
    if inter == Interpolation.BICUBIC:
        return cv2.INTER_CUBIC
    elif inter == Interpolation.LINEAR:
        return cv2.INTER_LINEAR
    elif inter == Interpolation.NEAREST:
        return cv2.INTER_NEAREST
    else:
        raise ValueError('Invalid interpolation type.')

class ElasticDistortion(Augmenter):
    def __init__(self, grid_width, grid_height, magnitude,
                 name=None, deterministic=False, random_state=None):

        super(ElasticDistortion, self).__init__(name=name,
                                                deterministic=deterministic,
                                                random_state=random_state)

        if isinstance(grid_width, StochasticParameter):
            self.grid_width = grid_width
        else:
            self.grid_width = handle_continuous_param(
                grid_width, 'grid_width', (0, None))

        if isinstance(grid_height, StochasticParameter):
            self.grid_height = grid_height
        else:
            self.grid_height = handle_continuous_param(
                grid_height, 'grid_height', (0, None))

        if isinstance(magnitude, StochasticParameter):
            self.magnitude = magnitude
        else:
            self.magnitude = handle_continuous_param(
                magnitude, 'magnitude', (0, None))

    def _gen_dimensions(self, img_size, h_tiles, v_tiles):
        w, h = img_size
        horizontal_tiles = h_tiles
        vertical_tiles = v_tiles

        width_of_square = int(math.floor(w / float(horizontal_tiles)))
        height_of_square = int(math.floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        return dimensions

    def _gen_polygons(self, dimensions, h_tiles, v_tiles, mag, random_state):
        horizontal_tiles = h_tiles
        vertical_tiles = v_tiles

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random_state.uniform(-mag, mag)
            dy = random_state.uniform(-mag, mag)
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                            x2, y2,
                            x3 + dx, y3 + dy,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                            x2 + dx, y2 + dy,
                            x3, y3,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                            x2, y2,
                            x3, y3,
                            x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                            x2, y2,
                            x3, y3,
                            x4, y4]

        return polygons

    def _elastic_distortion(self, img, interpolation, random_state):
        tile_w = self.grid_width.draw_sample(random_state)
        tile_h = self.grid_height.draw_sample(random_state)

        h_tiles = int(round(img.shape[0] / tile_h))
        v_tiles = int(round(img.shape[1] / tile_w))

        mag = self.magnitude.draw_sample(random_state)
        mag = int(round(mag * min(h_tiles, v_tiles)))

        if mag == 0:
            return img

        pil_img = Image.fromarray(img)

        # h_tiles = self.grid_width.draw_sample(random_state)
        # v_tiles = self.grid_height.draw_sample(random_state)
        # mag = self.magnitude.draw_sample(random_state)

        dimensions = self._gen_dimensions(pil_img.size, h_tiles, v_tiles)
        polygons = self._gen_polygons(dimensions, h_tiles, v_tiles,
                                      mag, random_state)

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        res = pil_img.transform(pil_img.size, Image.MESH, generated_mesh,
                                resample=_inter2pil(interpolation))

        return np.asarray(res)

    def _augment_images(self, images, random_state, parents, hooks):
        res_list = []
        for img in images:
            res = self._elastic_distortion(img, Interpolation.BICUBIC,
                                           random_state)
            res_list.append(res)

        return np.array(res_list)

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        res_list = []
        for segmap in segmaps:
            mask = segmap.get_arr()
            res = self._elastic_distortion(mask, Interpolation.NEAREST,
                                           random_state)
            res_list.append(SegmentationMapsOnImage(res, res.shape))

        return np.array(res_list)

    def get_parameters(self):
        return [self.grid_width, self.grid_height, self.magnitude]
