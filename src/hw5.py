import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(img):
    # 将图像从 BGR 转换为 HLS 色彩空间
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # 分离 H、L、S 三个通道
    h_channel, l_channel, s_channel = cv2.split(hls_img)
    
    # 保存原始 L 通道数据用于绘制 CDF
    original_l = l_channel.copy()
    
    # 对 L 通道进行直方图均衡化
    h, w = l_channel.shape

    # 手动计算原始图像的直方图和 CDF
    hist_original = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist_original[l_channel[i, j]] += 1
    cdf_original = hist_original.cumsum()
    cdf_original_normalized = cdf_original / cdf_original.max()  # 归一化到 [0,1]

    # 计算均衡化映射表
    hist = hist_original
    cdf = cdf_original
    cdf_normalized = cdf * 255 / cdf[-1]
    equalization_map = cdf_normalized.astype('uint8')

    # 应用映射表进行像素值映射
    equalized_l = np.zeros_like(l_channel)
    for i in range(h):
        for j in range(w):
            equalized_l[i, j] = equalization_map[l_channel[i, j]]
    
    # 手动计算均衡化后图像的直方图和 CDF
    hist_equalized = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist_equalized[equalized_l[i, j]] += 1
    cdf_equalized = hist_equalized.cumsum()
    cdf_equalized_normalized = cdf_equalized / cdf_equalized.max()  # 归一化到 [0,1]
        
    # 合并 H、均衡化后的 L 和 S 通道
    equalized_hls = cv2.merge([h_channel, equalized_l, s_channel])
    
    # 将图像从 HLS 转回 BGR 色彩空间
    equalized_img = cv2.cvtColor(equalized_hls, cv2.COLOR_HLS2BGR)
    
    # # 绘制原始和均衡化后的 L 通道 CDF
    # plt.figure(figsize=(12,6))

    # plt.subplot(2,2,1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('原始图像')
    # plt.axis('off')

    # plt.subplot(2,2,2)
    # plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    # plt.title('均衡化后图像')
    # plt.axis('off')

    # plt.subplot(2,2,3)
    # plt.plot(cdf_original_normalized, color='blue')
    # plt.title('原始 L 通道 CDF')

    # plt.subplot(2,2,4)
    # plt.plot(cdf_equalized_normalized, color='red')
    # plt.title('均衡化后 L 通道 CDF')

    # plt.tight_layout()
    # plt.show()
    
    return equalized_img

def clahe_algorithm(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # 将图像从 BGR 转换为 LAB 色彩空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 分离 L、A、B 三个通道
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    # 保存原始 L 通道用于绘制 CDF
    original_l = l_channel.copy()

    # 获取图像尺寸
    h, w = l_channel.shape

    # 计算每个 tile 的大小
    tile_h = h // tile_grid_size[1]
    tile_w = w // tile_grid_size[0]

    # 初始化空的数组用于存储均衡化的结果
    equalized_l = np.zeros_like(l_channel, dtype=np.float32)

    # 对每个 tile 进行处理
    for i in range(tile_grid_size[1]):
        for j in range(tile_grid_size[0]):
            # 计算 tile 的位置
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i != tile_grid_size[1] -1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j != tile_grid_size[0] -1 else w

            # 提取 tile
            tile = l_channel[y_start:y_end, x_start:x_end]

            # 计算直方图
            hist = np.bincount(tile.flatten(), minlength=256).astype(np.float32)

            # 计算剪辑阈值
            clip_limit_value = clip_limit * np.mean(hist)

            # 对直方图进行剪辑
            excess = hist - clip_limit_value
            excess[excess < 0] = 0
            hist = hist.clip(max=clip_limit_value)
            # 重新分配被剪辑的像素
            redistribute = excess.sum() / 256
            hist += redistribute

            # 计算剪辑后的 CDF
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf = cdf.astype('uint8')

            # 映射像素值
            equalized_tile = cdf[tile]

            # 将处理后的 tile 放回对应位置
            equalized_l[y_start:y_end, x_start:x_end] = equalized_tile

    # 进行双线性插值，平滑 tiles 之间的过渡
    # 创建网格坐标
    grid_x = np.linspace(0, w, tile_grid_size[0]+1)
    grid_y = np.linspace(0, h, tile_grid_size[1]+1)
    grid_x = grid_x.astype(int)
    grid_y = grid_y.astype(int)

    # 初始化空的数组用于存储插值结果
    final_l = np.zeros_like(l_channel, dtype=np.uint8)

    for i in range(tile_grid_size[1]):
        for j in range(tile_grid_size[0]):
            # 当前 tile 的四个角点
            x1, x2 = grid_x[j], grid_x[j+1]
            y1, y2 = grid_y[i], grid_y[i+1]

            # 获取当前 tile 及相邻 tiles 的中心值用于插值
            tile = equalized_l[y1:y2, x1:x2]

            # 插值
            final_l[y1:y2, x1:x2] = tile

    # 将 L 通道转换为 uint8 类型
    final_l = final_l.astype('uint8')

    # 计算均衡化后 L 通道的直方图和 CDF
    hist_equalized = np.bincount(final_l.flatten(), minlength=256).astype(np.float32)
    cdf_equalized = hist_equalized.cumsum()
    cdf_equalized_normalized = cdf_equalized / cdf_equalized.max()  # 归一化到 [0,1]

    # 合并均衡化后的 L 通道和原始的 A、B 通道
    equalized_lab = cv2.merge([final_l, a_channel, b_channel])

    # 将图像从 LAB 转回 BGR 色彩空间
    equalized_img = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2BGR)

    # 手动计算原始 L 通道的直方图和 CDF
    hist_original = np.bincount(original_l.flatten(), minlength=256).astype(np.float32)
    cdf_original = hist_original.cumsum()
    cdf_original_normalized = cdf_original / cdf_original.max()

    # 绘制原始和均衡化后的 L 通道 CDF
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    plt.title('手动 CLAHE 处理后图像')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(cdf_original_normalized, color='blue')
    plt.title('原始 L 通道 CDF')

    plt.subplot(2, 2, 4)
    plt.plot(cdf_equalized_normalized, color='red')
    plt.title('CLAHE 后 L 通道 CDF')

    plt.tight_layout()
    plt.show()

    return equalized_img

# 使用示例
if __name__ == "__main__":
    img = cv2.imread('./pic/hw5/2039.jpg')  # 确保使用彩色图像
    equalized_img = clahe_algorithm(img)
    # 如果需要保存均衡化后的图像，可以取消下面的注释
    # cv2.imshow('equalized_image', equalized_img)
    # cv2.waitKey(0)
    # cv2.imwrite('equalized_image.jpg', equalized_img)