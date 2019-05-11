#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import groupby
from typing import List

import numpy as np
from PIL import Image


def make_collage(input_paths: List[str], n_column: int, output_path: str):
    matrix_path = [[e[1] for e in g] for _, g in groupby(enumerate(input_paths), key=lambda x: x[0] // n_column)]
    matrix_images = [[Image.open(fpath) for fpath in row_path] for row_path in matrix_path]

    max_width = max((img for row_image in matrix_images for img in row_image), key=lambda img: img.size[0]).size[0]
    max_height = max((img for row_image in matrix_images for img in row_image), key=lambda img: img.size[1]).size[1]

    def padding_img(img: np.array, new_height: int, new_width: int, border_size: int) -> np.array:
        img_height, img_width, n_color_channel = img.shape
        top = (new_height - img_height) // 2
        left = (new_width - img_width) // 2
        # default is 1
        new_img = np.ones((new_height, new_width, 4), dtype=np.uint8) * 255

        if border_size > 0:
            default_val = [0, 0, 0, 170]

            new_img[:border_size, :, :] = default_val
            new_img[-border_size:, :, :] = default_val
            new_img[:, :border_size, :] = default_val
            new_img[:, -border_size:, :] = default_val

        new_img[top:top + img_height, left:left + img_width, :img.shape[-1]] = img
        return new_img

    def format_image(img: Image):
        border_size = 5
        return padding_img(np.asarray(img), max_height + 2 * border_size, max_width + 2 * border_size, border_size)

    collage = []
    for row_imgs in matrix_images:
        row_imgs = [format_image(img) for img in row_imgs]
        if len(row_imgs) < n_column:
            row_imgs = row_imgs + [np.ones(row_imgs[0].shape, dtype=np.uint8) * 255] * (n_column - len(row_imgs))
        collage.append(np.hstack(img for img in row_imgs))

    Image.fromarray(np.vstack(collage)).save(output_path)
