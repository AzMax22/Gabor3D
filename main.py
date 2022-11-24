import numpy as np
import cv2
import matplotlib.pyplot as plt  # package for plot function


class Gabor3D:
    def __init__(self, size_xy, size_deep, theta, w_t0, w, sigma_xy, sigma_t):
        """
        :param size_xy: размер фильтра по осям x и y, задается нечетным числом
        :param size_deep: размер фильтра по оси t, задается нечетным числом
        :param theta: угол поворота фильтра (в радианах)
        :param w_t0: частота по оси t
        :param w: пространственная частота
        :param sigma: ско функции Гаусса
        """

        sigma_x = sigma_xy
        sigma_y = sigma_xy

        radius_xy = size_xy // 2
        radius_deep = size_deep // 2
        array_axis_xy = range( -radius_xy, radius_xy + 1)
        array_axis_t = range(-radius_deep, radius_deep + 1)
        t, y, x = np.meshgrid(array_axis_t, array_axis_xy, array_axis_xy, indexing='ij')

        w_x0 = w * np.cos(theta)
        w_y0 = w * np.sin(theta)

        coef = 5 / ( (2 * np.pi)**(3/2) * sigma_t * sigma_x * sigma_y)  # коэффициент

        gauss_func = np.exp(- 0.5 * ((t**2)/(sigma_t**2) + (y**2)/(sigma_y**2) + (x**2)/(sigma_x**2)))  # функция гаусса

        arg = 2 * np.pi * (w_x0*x + w_y0*y + w_t0*t)  # аргумент cos или sin

        self.even_filter = (coef * gauss_func * np.cos(arg)).astype(np.float32)
        self.odd_filter = (coef * gauss_func * np.sin(arg)).astype(np.float32)



    def _convolution3D(self, img_blocks, filter3D):
        # предполагается что длина фильтра по z равна длине блока изб по z
        size_z = filter3D.shape[0]

        out_img = np.zeros_like(img_blocks[0], dtype=np.float32)

        for i in range(size_z):
            out_img = out_img + cv2.filter2D(img_blocks[i].astype(np.float32), -1, filter3D[i])

        return out_img

    def filtered(self, img_blocks):
        odd_img = self._convolution3D(img_blocks, self.odd_filter)
        even_img = self._convolution3D(img_blocks, self.even_filter)

        res_ing = odd_img**2 + even_img**2
        res_ing[res_ing > 255] = 255
        return res_ing.astype(np.uint8)


def show_filter3D(filter3D, size_ax_x, along_axis_show = "t"):
    if len(along_axis_show) == 1:
        _show_filter3D(filter3D, size_ax_x, along_axis_show)
        plt.show()
    else:
        for i, ch in enumerate(along_axis_show):
            _show_filter3D(filter3D, size_ax_x[i], ch)
        plt.show()


def _show_filter3D(filter3D, size_ax_x, along_axis_show = "t"):
    if along_axis_show == "t":
        len_filter = filter3D.shape[0]
    elif along_axis_show == "x":
        len_filter = filter3D.shape[2]
    elif along_axis_show == "y":
        len_filter = filter3D.shape[1]

    size_ax_y = len_filter // size_ax_x
    if len_filter % size_ax_x != 0:
        size_ax_y += 1

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Axis ' + along_axis_show)

    for i in range(len_filter):
        if along_axis_show == "t":
            g_2d = filter3D[i]
        elif along_axis_show == "x":
            g_2d = filter3D[:, :, i]
        elif along_axis_show == "y":
            g_2d = filter3D[:, i, :]

        plt.subplot(size_ax_y, size_ax_x, i + 1)
        plt.imshow(g_2d, cmap=plt.gray())
        plt.title(str(i))
        plt.xticks([]), plt.yticks([])
        print(f"g_2d(ax = {along_axis_show},n = {i}) = ", [np.min(g_2d), np.max(g_2d)], g_2d.dtype)


def show_images(list_img, names, size_ax_x):
    size_ax_y = len(list_img) // size_ax_x
    if len(list_img) % size_ax_x != 0:
        size_ax_y += 1

    for i in range(len(list_img)):
        img = list_img[i]
        plt.subplot(size_ax_x, size_ax_y, i+1)
        plt.imshow(img,  cmap=plt.gray())
        plt.title(names[i])
        plt.xticks([]), plt.yticks([])
        print(names[i], "=", [np.min(img), np.max(img)], img.dtype)
    plt.show()


def main():
    cap = cv2.VideoCapture("test.avi") #Video_37 test capture-2

    n = 7  # глубина блока изб и фильтра, должна быть нечетной

    block_img = []

    # создаем блок изб
    for i in range(n):
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block_img.append(gray_img)
    block_img = np.stack(block_img, axis=0)

    # параметры для банка фильтров
    list_w_t0 = [1/7, 1/8, 1/9]
    list_theta = [np.radians(0), np.radians(35), np.radians(75)]
    bank_filter = []

    # создание банка фильтров
    for w_t0 in list_w_t0:
        for theta in list_theta:
            gabor_filter = Gabor3D(size_xy=25, size_deep=n, theta=theta, w_t0=w_t0, w=1/4, sigma_xy=4, sigma_t=1)
            bank_filter.append(gabor_filter)


    while True:
        # применяем банк фильтров
        list_res_img = []
        for gabor_filter in bank_filter:
            res_img = gabor_filter.filtered(block_img)
            list_res_img.append(res_img)

        # удаление шума
        for img in list_res_img:
            std = np.std(img)
            img[img < std] = 0

        # объединение изображений после фильтрации
        list_res_img = np.stack(list_res_img, axis=0)
        count_nonzero = np.count_nonzero(list_res_img, axis=0)
        mask = count_nonzero > 4

        sum_ax_0 = np.sum(list_res_img, axis=0, where=mask)
        blobs = np.divide(sum_ax_0, count_nonzero, where=mask).astype(np.uint8)
        _, blobs = cv2.threshold(blobs, 80, 255, cv2.THRESH_BINARY)


        cv2.imshow("mask",blobs)


        # обновляем блок изб
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block_img = np.concatenate((np.delete(block_img, 0, 0), gray_img[None, :]))

        k = cv2.waitKey(1)

        if k == ord('w'):
            pass

        if k == ord('q'):
            break


if __name__ == '__main__':
    main()
