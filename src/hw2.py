import numpy as np
import cv2

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
        a = p[1]
        b = (p[2] - p[1])/2
        c = p[0] + 2*p[2] -(5*p[1] + p[3])/2
        d = (-p[0]+3*p[1]-3*p[2]+p[3])/2

        return a + b*x + c*x**2 + d*x**3
    
    for i in range(new_h):
        for j in range(new_w):
            x = i/scale
            y = j/scale
            x_0 = int(np.floor(x))
            y_0 = int(np.floor(y))
            dx = x - x_0
            dy = y - y_0

            patch = np.zeros((4, 4, 3), dtype=np.float32)
            for m in range(-1, 3):
                for n in range(-1, 3):
                    x_idx = np.clip(x_0 + m, 0, h-1)
                    y_idx = np.clip(y_0 + n, 0, w-1)
                    patch[m+1, n+1] = image[x_idx, y_idx]

            col = np.zeros((4, 3), dtype=np.float32)
            # for m in range(4):
            #     col[m] = cubic(patch[m], dy)
            for m in range(4):
                col[m] = cubic(patch[m], dy)
        
            new_image[i, j] = np.clip(cubic(col[:], dx), 0, 255)

    return new_image.astype(np.uint8)

if __name__ == '__main__':
    image = cv2.imread('../pic/logo.jpg')
    image = cv2.resize(image, (32, 32))
    cv2.imshow('original', cv2.resize(image, (256, 256)))
    # cv2.imshow('nearest', resize_nearest(image, 2))
    # cv2.imshow('bilinear', resize_bilinear(image, 2))
    cv2.imshow('bicubic', cv2.resize(resize_bicubic(image, 2), (256, 256)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()