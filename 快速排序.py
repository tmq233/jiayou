"""
快速排序（Quick Sort）

核心思想：分治（Divide and Conquer）
1. 从数组中选一个"基准"（pivot）
2. 把比基准小的放左边，比基准大的放右边（partition 分区）
3. 对左右两部分递归重复，直到每部分只剩 0 或 1 个元素

时间复杂度：平均 O(n log n)，最坏 O(n²)（数组已排序时，若总选最边上的元素为基准）
空间复杂度：O(log n)（递归栈深度）
稳定性：不稳定
"""


def partition(arr, low, high):
    """分区：把 arr[low..high] 按基准分成「小 | 基准 | 大」三部分，返回基准最终下标。

    这里采用 Lomuto 分区法（最容易理解）：
    - 选最右边元素 arr[high] 作为基准 pivot
    - 用指针 i 标记「小于等于基准」区域的末尾
    - 遍历 j，遇到 <= pivot 的元素就把它换到 i 的右边
    - 最后把 pivot 换到中间位置
    """
    pivot = arr[high]          # 选最右边元素为基准
    i = low - 1                # i：小于等于 pivot 的区域的末尾下标

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # 把基准放到「小于等于区域」的右边，此时基准左边全 <= 它，右边全 > 它
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quick_sort(arr, low, high):
    """对 arr[low..high] 区间做快速排序（原地修改）。"""
    if low >= high:            # 区间为空或只有一个元素，无需排序
        return
    pivot_index = partition(arr, low, high)
    quick_sort(arr, low, pivot_index - 1)   # 递归排左边
    quick_sort(arr, pivot_index + 1, high)  # 递归排右边


def sort(arr):
    """对外接口：返回排序后的新数组（不修改原数组）。"""
    result = arr[:]            # 拷贝一份，避免改动原数据
    quick_sort(result, 0, len(result) - 1)
    return result


if __name__ == "__main__":
    test_cases = [
        [3, 6, 8, 10, 1, 2, 1],   # 一般情况（含重复元素）
        [5, 4, 3, 2, 1],          # 逆序（最坏情况）
        [1, 2, 3, 4, 5],          # 已排序
        [42],                     # 单元素
        [],                       # 空数组
    ]
    for case in test_cases:
        print(f"原数组: {case}")
        print(f"排序后: {sort(case)}")
        print("-" * 30)
