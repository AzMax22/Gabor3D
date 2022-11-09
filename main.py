import numpy as np
import cv2
import matplotlib.pyplot as plt # package for plot function


class Gabor3D:
    def __init__(self, size, theta, w_t0, w, sigma):
        """
        :param size: размер фильтра (по всем осям одинаков), задается нечетным числом
        :param theta: угол поворота фильтра (в радианах)
        :param w_t0: частота по оси t ??????????
        :param w: пространственная частота  ????????????
        :param sigma: ско функции Гаусса ?????????????? (по идее зависит от size)
        """
        sigma_x = 2/3 * sigma
        sigma_y =  sigma
        sigma_t = 2/3 * sigma

        radius = size // 2
        array_axis = range( -radius, radius + 1)
        x, y, t = np.meshgrid(array_axis, array_axis, array_axis)

        w_x0 = w * np.cos(theta)
        w_y0 = w * np.sin(theta)

        coef = 1 / ( (2 * np.pi)**(3/2) * sigma_t * sigma_x * sigma_y)  # коэффициент

        gauss_func = np.exp(- 0.5 * ((t**2)/(sigma_t**2) + (y**2)/(sigma_y**2) + (x**2)/(sigma_x**2)))  # функция гаусса

        arg = 2 * np.pi * (w_x0*x + w_y0*y + w_t0*t)  # аргумент cos или sin

        self.odd_filter = coef * gauss_func * np.sin(arg)
        self.even_filter = coef * gauss_func * np.cos(arg)

    def _convolution3D(self, img_blocks, filter3D):
        # предполагается что длина фильтра по z равна длине блока изб по z
        size_z = filter3D.shape[2]

        out_img = np.zeros_like(img_blocks[:, :, 0])

        for i in range(size_z):
            out_img = out_img + cv2.filter2D(img_blocks[:, :, i], -1, filter3D[:, :, i])

        return out_img

    def filtered(self, img_blocks):
        odd_img = self._convolution3D(img_blocks, self.odd_filter)
        even_img = self._convolution3D(img_blocks, self.even_filter)

        return odd_img**2 + even_img**2






def show_filter3D(filter3D, along_axis_show = "t"):
    size_filter = filter3D.shape[0]
    size_ax_x = 5
    size_ax_y = size_filter // size_ax_x  + 1
    fig, axes = plt.subplots(size_ax_y, size_ax_x)

    i = 0
    for i in range(size_filter):
        y = i // size_ax_x
        x = i % size_ax_x

        if along_axis_show == "t":
            g_2d = filter3D[:, :, i]
        elif along_axis_show == "x":
            g_2d = filter3D[:, i, :]
        elif along_axis_show == "y":
            g_2d = filter3D[i, :, :]

        ax = axes[y][x]
        ax.imshow(g_2d, cmap=plt.gray())
        ax.set_title(str(i))
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i+1, size_ax_y * size_ax_x):
        y = j // size_ax_x
        x = j % size_ax_x
        fig.delaxes(axes[y][x])

    fig.set_figwidth(12)  # ширина и
    fig.set_figheight(6)  # высота "Figure"

    plt.show()


def show_images(list_img, names, size_ax_x):
    size_ax_y = len(list_img) // size_ax_x
    if len(list_img) % size_ax_x != 0 :
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

    cap = cv2.VideoCapture("test.avi")

    n = 5  # глубина блока изб и фильтра, должна быть нечетной

    ret, img = cap.read()
    block_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # создаем блок изб
    for i in range(n - 1):
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block_img = np.dstack((block_img, gray_img))

    gabor_filter = Gabor3D(size=n, theta=0, w_t0=0.2, w=0.2, sigma=0.3)

    show_filter3D(gabor_filter.odd_filter, "t")

    while True:

        res_img = gabor_filter.filtered(block_img)

        print([np.min(res_img), np.max(res_img)], res_img.dtype)
        cv2.imshow("res", res_img)
        #new = (res_img * (255 / np.max(res_img))).astype(np.uint8)
        #cv2.imshow("lin_res", new)

        # обновляем блок изб
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block_img = np.dstack((np.delete(block_img, 0, 2), gray_img))

        """# параметры для банка фильтров
        w = 0.4
        sigma = 0.5
        list_w_t0 = [0.15, 0.3, 0.45]
        list_theta = [0, np.pi/3, 2*np.pi/3]
        bank_filter = []
    
        # создание банка фильтров
        for w_t0 in list_w_t0:
            for theta in list_theta:
                gabor_filter = Gabor3D(size=n, theta=theta, w_t0=w_t0, w=w, sigma=sigma)
                bank_filter.append(gabor_filter)
    
        show_filter3D(bank_filter[2].odd_filter, "y")
    
        # применяем банк фильтров
        list_res_img = []
        for gabor_filter in bank_filter:
            res_img = gabor_filter.filtered(block_img)
            list_res_img.append(res_img)
    
        names = [f"theta_{j+1}, w_t0_{i+1}" for i in range(3) for j in range(3)]
        show_images(list_res_img, names, 3)
    
        #cv2.imshow("1", list_res_img[1])"""

        k = cv2.waitKey(1)

        if k == ord('w'):
            pass

        if k == ord('q'):
            break



if __name__ == '__main__':
    main()
