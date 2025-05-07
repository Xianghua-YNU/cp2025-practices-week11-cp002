import numpy as np
import matplotlib

# 设置 matplotlib 的后端为非交互式后端，这样在没有图形界面的环境中也能正常保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 定义物理常数：玻尔兹曼常数，单位为焦耳每开尔文（J/K）
kB = 1.380649e-23

# 样本参数
# 体积，将 1000 立方厘米转换为立方米
V = 1000e-6
# 原子数密度，单位为每立方米（m^-3）
rho = 6.022e28
# 德拜温度，单位为开尔文（K），是德拜模型中的一个关键参数
theta_D = 428

def integrand(x):
    """
    该函数用于计算德拜模型积分中的被积函数值。
    被积函数的表达式为：x^4 * e^x / (e^x - 1)^2
    :param x: 积分变量
    :return: 被积函数在 x 处的值
    """
    # 计算 e 的 x 次幂
    exp_x = np.exp(x)
    # 按照被积函数的表达式进行计算并返回结果
    return x ** 4 * exp_x / (exp_x - 1) ** 2

def gauss_quadrature(f, a, b, n):
    """
    此函数实现了高斯 - 勒让德积分法，用于数值计算定积分。
    :param f: 被积函数，是一个可调用对象
    :param a: 积分下限
    :param b: 积分上限
    :param n: 高斯点的数量，用于控制积分的精度
    :return: 定积分的近似值
    """
    # 使用 numpy 的 leggauss 函数获取高斯 - 勒让德求积的节点和对应的权重
    x, w = np.polynomial.legendre.leggauss(n)
    # 将 [-1, 1] 区间上的节点映射到 [a, b] 积分区间上
    t = 0.5 * (x + 1) * (b - a) + a
    # 根据高斯 - 勒让德积分公式计算积分值并返回
    return 0.5 * (b - a) * np.sum(w * f(t))

def cv(T):
    """
    该函数根据德拜模型计算给定温度 T 下固体的热容。
    :param T: 温度，单位为开尔文（K）
    :return: 热容值，单位为焦耳每开尔文（J/K）
    """
    # 当温度为 0 时，根据物理原理热容为 0
    if T == 0:
        return 0
    # 计算积分上限，与德拜温度和当前温度有关
    upper_limit = theta_D / T
    # 使用高斯 - 勒让德积分计算积分值
    integral = gauss_quadrature(integrand, 0, upper_limit, 50)
    # 根据德拜模型的热容公式计算热容值并返回
    return 9 * V * rho * kB * (T / theta_D) ** 3 * integral

def plot_cv():
    """
    该函数用于绘制热容随温度的变化曲线，并将绘制好的图片保存到指定路径。
    """
    # 生成从 5 到 500 开尔文的 200 个均匀分布的温度点
    T = np.linspace(5, 500, 200)
    # 计算每个温度点对应的热容值，并将结果存储为 numpy 数组
    C_V = np.array([cv(t) for t in T])

    # 创建一个新的图形窗口，设置图形的大小为 10 英寸宽，6 英寸高
    plt.figure(figsize=(10, 6))
    # 绘制热容随温度变化的曲线，使用蓝色实线，标签为 'Debye Model'
    plt.plot(T, C_V, 'b-', label='Debye Model')

    # 添加参考线，用于展示低温下的 T³ 定律
    # 生成从 5 到 50 开尔文的 50 个均匀分布的低温点
    T_low = np.linspace(5, 50, 50)
    # 根据 T³ 定律计算低温下的热容值
    C_low = cv(50) * (T_low / 50) ** 3
    # 绘制低温下的热容曲线，使用红色虚线，标签为 'T³ Law'
    plt.plot(T_low, C_low, 'r--', label='T³ Law')

    # 添加 x 轴标签，说明 x 轴表示的是温度，单位为开尔文（K）
    plt.xlabel('Temperature (K)')
    # 添加 y 轴标签，说明 y 轴表示的是热容，单位为焦耳每开尔文（J/K）
    plt.ylabel('Heat Capacity (J/K)')
    # 添加图表标题，表明该图表展示的是德拜模型下固体热容与温度的关系
    plt.title('Solid Heat Capacity vs Temperature (Debye Model)')
    # 添加网格线，方便观察图表中的数据点，设置网格线的透明度为 0.2
    plt.grid(True, which='both', ls='-', alpha=0.2)
    # 添加图例，用于说明不同曲线所代表的含义
    plt.legend()

    # 定义图片的保存路径
    save_path = r"C:\Users\31025\OneDrive\桌面\t\cv_plot.png"
    # 获取保存路径的目录部分
    save_dir = os.path.dirname(save_path)
    # 如果保存目录不存在，则创建该目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 将绘制好的图表保存为图片，设置图片的分辨率为 300 dpi
    plt.savefig(save_path, dpi=300)
    # 打印图片保存的信息，告知用户图片保存的具体路径
    print(f"Plot saved to: {save_path}")

def main():
    # 定义要测试的特征温度点
    test_temperatures = [5, 100, 300, 500]
    # 打印提示信息，表明接下来要测试不同温度下的热容值
    print("\n测试不同温度下的热容值：")
    # 打印分隔线，用于美化输出格式
    print("-" * 40)
    # 打印表头，说明第一列是温度，第二列是热容
    print("温度 (K)\t热容 (J/K)")
    # 打印分隔线，用于美化输出格式
    print("-" * 40)
    # 遍历每个测试温度点
    for T in test_temperatures:
        # 调用 cv 函数计算该温度下的热容值
        result = cv(T)
        # 按照指定格式打印温度和对应的热容值
        print(f"{T:8.1f}\t{result:10.3e}")

    # 调用 plot_cv 函数绘制热容随温度的变化曲线并保存图片
    plot_cv()

# 当脚本作为主程序运行时，调用 main 函数
if __name__ == '__main__':
    main()
