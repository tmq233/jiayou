"""Matplotlib Chinese font configuration."""
import matplotlib.pyplot as plt
import matplotlib
import platform


def setup_chinese_fonts():
    """Configure matplotlib to display Chinese characters correctly."""
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    elif platform.system() == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Heiti SC', 'PingFang SC', 'Apple SD Gothic Neo']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
