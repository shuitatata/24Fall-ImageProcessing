import numpy as np
import cv2
from numpy.linalg import LinAlgError

def resize_nearest(image, scale):
    """
    Nearest neighbor interpolation
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            origin_i = round(i/scale)
            origin_j = round(j/scale)
            origin_i = min(origin_i, h-1)
            origin_j = min(origin_j, w-1)
            new_image[i, j] = image[origin_i, origin_j]
    return new_image

def resize_bilinear(image, scale):
    """
    Bilinear interpolation
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)

    # new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    # for i in range(new_h):
    #     for j in range(new_w):
    #         origin_i = i/scale
    #         origin_j = j/scale
    #         x1, y1 = int(origin_i), int(origin_j)
    #         x2, y2 = min(x1+1, h-1), min(y1+1, w-1)
    #         u, v = origin_i-x1, origin_j-y1
    #         new_image[i, j] = (1-u)*(1-v)*image[x1, y1] + u*(1-v)*image[x2, y1] + (1-u)*v*image[x1, y2] + u*v*image[x2, y2]

    row_indices = np.arange(new_h).reshape(-1, 1).repeat(new_w, axis=1) / scale
    col_indices = np.arange(new_w).reshape(1, -1).repeat(new_h, axis=0) / scale
    
    x1 = np.floor(row_indices).astype(int)
    y1 = np.floor(col_indices).astype(int)
    x2 = np.clip(x1 + 1, 0, h - 1)
    y2 = np.clip(y1 + 1, 0, w - 1)
    
    u = row_indices - x1
    v = col_indices - y1

    # 将u复制为三维
    u = np.expand_dims(u, axis=-1)
    v = np.expand_dims(v, axis=-1)

    print(u.shape)

    new_image = (1 - u) * (1 - v) * image[x1, y1] + \
                u * (1 - v) * image[x2, y1] + \
                (1 - u) * v * image[x1, y2] + \
                u * v * image[x2, y2]
    
    return new_image.astype(np.uint8)

def resize_bicubic(image, scale):
    """
    Bicubic interpolation
    """
    
    h, w = image.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    def cubic(p:list, x):
        p = p.astype(np.float32)
        a = p[..., 1, :]
        b = (p[..., 2, :] - p[..., 0, :]) / 2
        c = p[..., 0, :] + 2 * p[..., 2, :] - (5 * p[..., 1, :] + p[..., 3, :]) / 2
        d = (-p[..., 0, :] + 3 * p[..., 1, :] - 3 * p[..., 2, :] + p[..., 3, :]) / 2
        return a + b * x + c * x ** 2 + d * x ** 3
    
    x_new = np.linspace(0, h-1, new_h)
    y_new = np.linspace(0, w-1, new_w)
    x, y = np.meshgrid(x_new, y_new, indexing='ij')

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = x - x0
    dy = y - y0

    x_indices = np.clip(np.add.outer(x0, np.arange(-1, 3)), 0, h-1)
    y_indices = np.clip(np.add.outer(y0, np.arange(-1, 3)), 0, w-1)

    print(x_indices.shape, y_indices.shape)

    patch = image[x_indices[:, :, :, None], y_indices[:, :, None, :], :]

    print(patch.shape)

    col = np.zeros((new_h, new_w, 4, 3), dtype=np.float32)
    for m in range(4):
        print(patch[:, :, m, :, :].shape)
        col[:, :, m, :] = cubic(patch[:, :, m, :, :], dy[:, :, None])
    
    new_image = np.clip(cubic(col, dx[:, :, None]), 0, 255)

    return new_image.astype(np.uint8)

def get_rotation_matrix(angle):
    angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

def get_translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

def get_shear_matrix(kx,ky):
    return np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]])

def apply_transform(image, matrix, output_shape=None):
    """
    Apply a geometric transformation to an image using a given matrix.
    Parameters:
        - image: Input image.
        - matrix: 2x2 transformation matrix.
        - output_shape: The desired shape (height, width) of the output image.
    Returns:
        - Transformed image.
    """
    h, w, c = image.shape
    if output_shape is None:
        output_shape = (h, w)
    new_h, new_w = output_shape
    new_image = np.zeros((new_h, new_w, c), dtype=np.uint8)

    def valid_coord(x, y):
        return (0 <= x) & (x < h) & (0 <= y) & (y < w)

    # 生成目标图像的坐标网格
    x, y = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')  # Shape: (new_h, new_w)
    coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3).astype(np.float32)  # Shape: (new_h*new_w, 2)

    # 计算逆变换矩阵
    try:
        inverse_matrix = np.linalg.inv(matrix)
    except LinAlgError:
        inverse_matrix = matrix
    print(inverse_matrix.shape)

    # 计算在原图中的坐标（逆映射）
    original_coords = coords @ inverse_matrix.T
    
    original_coords = original_coords.reshape(new_h, new_w, 3)
    print(image.shape, new_h)

    # 分解为浮点和整数坐标
    x_orig = original_coords[:, :, 0] / original_coords[:, :, 2]
    y_orig = original_coords[:, :, 1] / original_coords[:, :, 2]
    x1 = np.floor(x_orig).astype(int)
    y1 = np.floor(y_orig).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1
    # x1 = np.clip(x1, 0, h - 1)
    # y1 = np.clip(y1, 0, w - 1)
    # x2 = np.clip(x1 + 1, 0, h - 1)
    # y2 = np.clip(y1 + 1, 0, w - 1)
    
    # 计算插值权重
    u = (x_orig - x1)[..., None]  # Shape: (new_h, new_w, 1)
    v = (y_orig - y1)[..., None]  # Shape: (new_h, new_w, 1)

    # 使用双线性插值
    x1_valid = np.clip(x1, 0, h - 1)
    y1_valid = np.clip(y1, 0, w - 1)
    x2_valid = np.clip(x1 + 1, 0, h - 1)
    y2_valid = np.clip(y1 + 1, 0, w - 1)
    print(image.shape)
    print(x1.shape)
    new_image = ((1 - u) * (1 - v) * np.where(valid_coord(x1, y1)[..., None], image[x1_valid, y1_valid], 0) +
                 u * (1 - v) * np.where(valid_coord(x1, y2)[..., None], image[x1_valid, y2_valid], 0) +
                 (1 - u) * v * np.where(valid_coord(x2, y1)[..., None], image[x2_valid, y1_valid], 0) +
                 u * v * np.where(valid_coord(x2, y2)[..., None], image[x2_valid, y2_valid], 0))

    return new_image.astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread('../pic/flower.jpg')
    image = cv2.resize(image, (256, 256))
    # 绘制图像边框
    image[:5, :, :] = 255
    image[-5:, :, :] = 255
    image[:, :5, :] = 255
    image[:, -5:, :] = 255
    cv2.imshow('original', image)
    # cv2.imshow('nearest', resize_nearest(image, 2))
    nearest = resize_nearest(image, 4)
    bilinear = resize_bilinear(image, 4)
    bicubic = resize_bicubic(image, 4)

    cv2.imwrite('../assest/hw2/flower_nearest.jpg', nearest)
    cv2.imwrite('../assest/hw2/flower_bilinear.jpg', bilinear)
    cv2.imwrite('../assest/hw2/flower_bicubic.jpg', bicubic)

    nearest_cv2 = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    bilinear_cv2 = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    bicubic_cv2 = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite('../assest/hw2/flower_nearest_cv2.jpg', nearest_cv2)
    cv2.imwrite('../assest/hw2/flower_bilinear_cv2.jpg', bilinear_cv2)
    cv2.imwrite('../assest/hw2/flower_bicubic_cv2.jpg', bicubic_cv2)

    # print(np.allclose(bilinear, bicubic))
    # cv2.imshow('bilinear', bilinear)
    # cv2.imshow('bicubic', bicubic)

    # rotation_matrix = get_rotation_matrix(0)
    # translation_matrix = get_translation_matrix(0, 0)
    # translation_matrix2 = get_translation_matrix(0, 0)

    # transform_matrix = translation_matrix2 @ rotation_matrix @ translation_matrix
    # transform_matrix = get_shear_matrix(0.5, 0.5)
    # new_image = apply_transform(image, transform_matrix)
    # cv2.imshow('rotated', new_image)
    # cv2.imwrite('../assest/hw2/flower_sheared.jpg', new_image)
    cv2.imwrite('../assest/hw2/flower_low.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()