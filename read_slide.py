import os
import numpy as np
from PIL import Image
import cv2
from multiprocessing import Pool


def apply_top_hat(image, kernel_size):
    """Apply a top-hat morphological transformation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    tophat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat_img


def is_edge(frame_id):
    if frame_id <= 48 or frame_id > 2256 or frame_id % 24 in {0, 1, 2, 23}:
        return True
    else:
        return False


def read_image_to_numpy_array(filename):
    try:
        im = Image.open(filename + ".jpg")
        ans = np.array(im)
        im.close()
    except FileNotFoundError:
        im = Image.open(filename + ".tif")
        ans = np.array(im)
        im.close()
    except:
        ans = np.zeros(shape=(1004, 1362))

    return apply_top_hat(ans, kernel_size=(75, 75))


def get_tile_num(frame_id):
    if frame_id < 10:
        return "0000" + str(frame_id)
    elif frame_id < 100:
        return "000" + str(frame_id)
    elif frame_id < 1000:
        return "00" + str(frame_id)
    elif frame_id < 10000:
        return "0" + str(frame_id)
    else:
        return str(frame_id)


def create_minitiles_optimized(image):
    tile_size = 32
    overlap = 1
    stride = tile_size - overlap

    # Determine number of tiles along each dimension
    n_tiles_y = (image.shape[0] - tile_size) // stride + 1
    n_tiles_x = (image.shape[1] - tile_size) // stride + 1

    # Preallocate array for minitiles
    minitiles = np.zeros(
        (n_tiles_y * n_tiles_x, tile_size, tile_size, 1), dtype=image.dtype
    )

    # Use vectorized operations to compute all tiles
    index = 0
    for y in range(n_tiles_y):
        for x in range(n_tiles_x):
            minitile = image[
                y * stride : y * stride + tile_size, x * stride : x * stride + tile_size
            ]
            minitiles[index, :, :, 0] = minitile
            index += 1

    return minitiles


def create_minitiles(image):
    """
    image: type = numpy.array, shape = (1004,1362)
    returns: array of minitiles, type: numpy.array, shape=(1376,32,32,1) 32*32 + 20 overlaps of 1,
        32*43+14 overlaps of 1
    """

    n_tiles_y = 32
    n_tiles_x = 43

    minitiles = np.zeros((n_tiles_y * n_tiles_x, 32, 32, 1), dtype=image.dtype)

    idx = 0
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            check_i = i < 15
            check_j = j < 21

            if check_i and check_j:
                xstart, xend, ystart, yend = (
                    i * 31,
                    (i * 31 + 32),
                    j * 31,
                    (j * 31 + 32),
                )
            elif check_i and not check_j:
                xstart, xend, ystart, yend = (
                    i * 31,
                    (i * 31 + 32),
                    (j - 21) * 32 + 652,
                    ((j - 21) * 32 + 32 + 652),
                )
            elif not check_i and check_j:
                xstart, xend, ystart, yend = (
                    (i - 15) * 32 + 466,
                    ((i - 15) * 32 + 32 + 466),
                    j * 31,
                    (j * 31 + 32),
                )
            else:
                xstart, xend, ystart, yend = (
                    (i - 15) * 32 + 466,
                    ((i - 15) * 32 + 32 + 466),
                    (j - 21) * 32 + 652,
                    ((j - 21) * 32 + 32 + 652),
                )

            minitiles[idx, :, :, 0] = image[ystart:yend, xstart:xend]
            idx += 1

    return minitiles


def create_tile_dataset_range(slide_id, start_frame, end_frame, slide_directory):

    directory = slide_directory + "/" + slide_id + "/"
    minitile_list = []

    for i in range(
        start_frame, end_frame
    ):  # for tomorrow maybe you could stop at a smaller number if this one gives you memory issues.

        if not is_edge(i):  ### This is the function that Amin shared.

            tile_c1 = read_image_to_numpy_array(
                directory + "Tile0" + get_tile_num(i)
            )  # This reads the tile file for channel 1
            tile_c2 = read_image_to_numpy_array(
                directory + "Tile00" + str(i + 2304)
            )  # This reads the tile file for channel 2
            tile_c3 = read_image_to_numpy_array(
                directory + "Tile00" + str(i + 4608)
            )  # This reads the tile file for channel 3
            tile_c4 = read_image_to_numpy_array(
                directory + "Tile0" + get_tile_num(i + 9216)
            )  # This reads the tile file for channel 4

            minitiles_c1 = create_minitiles(tile_c1)  # function above
            minitiles_c2 = create_minitiles(tile_c2)  # function above
            minitiles_c3 = create_minitiles(tile_c3)  # function above
            minitiles_c4 = create_minitiles(tile_c4)  # function above

            minitiles_allchannels = np.concatenate(
                [minitiles_c1, minitiles_c2, minitiles_c3, minitiles_c4], axis=3
            )

            minitile_list.append(minitiles_allchannels)

    all_minitiles = np.concatenate(minitile_list, axis=0).astype(np.uint8)

    return all_minitiles


def create_tile_dataset(slide_id, slide_directory):
    with Pool(processes=8) as pool:
        # create a list of arguments for each function call
        args = [(slide_id, i, i + 100, slide_directory) for i in range(1, 2301, 100)]
        # use the pool to process the tiles in parallel
        results = pool.starmap(create_tile_dataset_range, args)
    return np.concatenate(results, axis=0).astype(np.uint8)
