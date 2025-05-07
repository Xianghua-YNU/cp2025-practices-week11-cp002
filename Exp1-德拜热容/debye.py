import numpy as np
import matplotlib

matplotlib.use('Agg')  # 设置后端为非交互式后端
import matplotlib.pyplot as plt
import os

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K


def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2"""
    exp_x = np.exp(x)
    return x ** 4 * exp_x / (exp_x - 1) ** 2


def gauss_quadrature(f, a, b, n):
    """实现高斯-勒让德积分"""
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * f(t))


def cv(T):
    """计算给定温度T下的热容"""
    if T == 0:
        return 0
    upper_limit = theta_D / T
    integral = gauss_quadrature(integrand, 0, upper_limit, 50)
    return 9 * V * rho * kB * (T / theta_D) ** 3 * integral


def plot_cv():
    """绘制热容随温度的变化曲线，并保存图片"""
    T = np.linspace(5, 500, 200)
    C_V = np.array([cv(t) for t in T])

    plt.figure(figsize=(10, 6))
    plt.plot(T, C_V, 'b-', label='Debye Model')

    # 添加参考线
    T_low = np.linspace(5, 50, 50)
    C_low = cv(50) * (T_low / 50) ** 3
    plt.plot(T_low, C_low, 'r--', label='T³ Law')

    # 添加标签和标题
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Capacity (J/K)')
    plt.title('Solid Heat Capacity vs Temperature (Debye Model)')
    plt.grid(True, which='both', ls='-', alpha=0.2)
    plt.legend()

    # 保存图片到指定路径
    save_path = r"C:\Users\31025\OneDrive\桌面\t\cv_plot.png"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")


def main():
    # 测试一些特征温度点的热容值
    test_temperatures = [5, 100, 300, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)")
    print("-" * 40)
    for T in test_temperatures:
        result = cv(T)
        print(f"{T:8.1f}\t{result:10.3e}")

    # 绘制热容曲线
    plot_cv()


if __name__ == '__main__':
    main()
