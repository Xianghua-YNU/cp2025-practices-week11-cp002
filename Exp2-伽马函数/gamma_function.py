import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial, sqrt, pi

# --- Task 1: 绘制被积函数 ---

def integrand_gamma(x, a):
    """
    伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)
    使用对数技巧提高数值稳定性，尤其当 x 或 a 较大时。
    f = exp((a-1)*log(x) - x)
    """
    # 处理 x=0 的情况
    if x == 0:
        if a > 1:
            return 0.0
        elif a == 1:
            return 1.0
        else:
            return np.inf
    elif x > 0:
        try:
            log_f = (a - 1) * np.log(x) - x
            return np.exp(log_f)
        except ValueError:
            return 0.0
    else:
        return 0.0

def plot_integrands():
    """绘制 a=2, 3, 4 时的被积函数图像"""
    x_vals = np.linspace(0.01, 10, 400)
    plt.figure(figsize=(10, 6))

    for a_val in [2, 3, 4]:
        y_vals = np.array([integrand_gamma(x, a_val) for x in x_vals])
        valid_indices = np.isfinite(y_vals)
        plt.plot(x_vals[valid_indices], y_vals[valid_indices], label=f'$a = {a_val}$')

        peak_x = a_val - 1
        if peak_x > 0:
            peak_y = integrand_gamma(peak_x, a_val)
            plt.plot(peak_x, peak_y, 'o', ms=5, label=f'Peak at x={peak_x}' if a_val == 2 else None)

    plt.xlabel("$x$")
    plt.ylabel("$f(x, a) = x^{a-1} e^{-x}$")
    plt.title("Integrand of the Gamma Function")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.show()

# --- Task 2 & 3: 解析推导 ---
# Task 2: 峰值位置推导见注释
# Task 3: 变量代换推导见注释

# --- Task 4: 实现伽马函数计算 ---

def transformed_integrand_gamma(z, a):
    """
    变换后的被积函数 g(z, a) = f(x(z), a) * dx/dz
    其中 x = cz / (1-z) 和 dx/dz = c / (1-z)^2, 且 c = a-1
    假设 a > 1
    """
    c = a - 1.0
    if c <= 0:
        return 0.0

    if z < 0 or z > 1:
        return 0.0
    if z == 1:
        return 0.0

    x = c * z / (1.0 - z)
    dxdz = c / ((1.0 - z)**2)
    val_f = integrand_gamma(x, a)

    if not np.isfinite(val_f) or not np.isfinite(dxdz):
        return 0.0

    return val_f * dxdz

def gamma_function(a):
    """
    计算 Gamma(a)
    - 如果 a > 1, 使用变量代换 z = x/(c+x) 和 c=a-1 进行数值积分。
    - 如果 a <= 1, 直接对原始被积函数进行积分。
    """
    if a <= 0:
        print(f"警告: Gamma(a) 对 a={a} <= 0 无定义。")
        return np.nan

    try:
        if a > 1.0:
            result, error = quad(transformed_integrand_gamma, 0, 1, args=(a,))
        else:
            result, error = quad(integrand_gamma, 0, np.inf, args=(a,))
        return result
    except Exception as e:
        print(f"计算 Gamma({a}) 时发生错误: {e}")
        return np.nan

# --- 主程序 ---
def test_gamma():
    """测试伽马函数的计算结果"""
    a_test = 1.5
    result = gamma_function(a_test)
    expected = np.sqrt(np.pi) / 2
    relative_error = abs(result - expected) / expected if expected != 0 else 0
    print(f"Γ({a_test}) = {result:.8f} (精确值: {expected:.8f}, 相对误差: {relative_error:.2e})")

    test_values = [3, 6, 10]
    print("\n测试整数值：")
    print("-" * 60)
    print("a\t计算值 Γ(a)\t精确值 (a-1)!\t相对误差")
    print("-" * 60)
    for a in test_values:
        result = gamma_function(a)
        factorial_val = float(factorial(a-1))
        relative_error = abs(result - factorial_val) / factorial_val if factorial_val != 0 else 0
        print(f"{a}\t{result:<12.6e}\t{factorial_val:<12.0f}\t{relative_error:.2e}")
    print("-" * 60)

def main():
    plot_integrands()
    test_gamma()

if __name__ == '__main__':
    main()
