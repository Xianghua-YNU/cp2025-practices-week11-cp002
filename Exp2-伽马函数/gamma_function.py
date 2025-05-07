import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式后端
import matplotlib.pyplot as plt
import os

def integrand_gamma(x, a):
    """计算伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)"""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0 if a > 1 else np.inf
    return x**(a-1) * np.exp(-x)

def plot_integrands():
    """绘制 a=2, 3, 4 时的被积函数图像"""
    x_vals = np.linspace(0.01, 10, 400)  # 从略大于0开始
    plt.figure(figsize=(10, 6))

    for a_val in [2, 3, 4]:
        y_vals = [integrand_gamma(x, a_val) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f'$a = {a_val}$')

    plt.xlabel("$x$")
    plt.ylabel("$f(x, a) = x^{a-1} e^{-x}$")
    plt.title("Integrand of the Gamma Function")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    # 保存图片到指定路径
    save_path = r"C:\Users\31025\OneDrive\桌面\t\gamma_integrand_plot.png"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    plot_integrands()
