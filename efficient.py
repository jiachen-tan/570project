import time
import sys

# 尝试导入 resource (Linux 环境下用于测内存，这是评分环境的标准)
try:
    import resource
except ImportError:
    resource = None

# 尝试导入 psutil (Windows 环境下用于测内存，可选)
try:
    import psutil
except ImportError:
    psutil = None


class InputGenerator:
    """
    负责读取输入文件并生成最终的两个字符串。
    逻辑与 basic.py 完全一致。
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def generate(self):
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # 解析第一个字符串
        base_s1 = lines[0]
        indices_s1 = []
        current_idx = 1

        # 读取第一个字符串的生成规则
        while current_idx < len(lines):
            if lines[current_idx].isdigit():
                indices_s1.append(int(lines[current_idx]))
                current_idx += 1
            else:
                break

        # 解析第二个字符串
        if current_idx < len(lines):
            base_s2 = lines[current_idx]
            indices_s2 = []
            current_idx += 1
            while current_idx < len(lines):
                if lines[current_idx].isdigit():
                    indices_s2.append(int(lines[current_idx]))
                    current_idx += 1
                else:
                    break
        else:
            base_s2 = ""
            indices_s2 = []

        return self._process_string(base_s1, indices_s1), self._process_string(base_s2, indices_s2)

    def _process_string(self, base_str, indices):
        """
        根据题目规则生成字符串: S = S[:n+1] + S + S[n+1:]
        """
        s = base_str
        for idx in indices:
            idx = int(idx)
            s = s[:idx + 1] + s + s[idx + 1:]
        return s


class EfficientSolver:
    """
    实现 Hirschberg 算法 (分治 + 空间优化 DP)
    """

    def __init__(self):
        # 题目规定的罚分参数
        self.DELTA = 30
        self.ALPHA = {
            'A': {'A': 0, 'C': 110, 'G': 48, 'T': 94},
            'C': {'A': 110, 'C': 0, 'G': 118, 'T': 48},
            'G': {'A': 48, 'C': 118, 'G': 0, 'T': 110},
            'T': {'A': 94, 'C': 48, 'G': 110, 'T': 0}
        }

    def get_last_row_score(self, s1, s2):
        """
        空间优化版 DP：只存储两行，计算 s1 和 s2 对齐后的最后一行 DP 值。
        空间复杂度: O(len(s2))，即线性空间。
        """
        m = len(s1)
        n = len(s2)

        # prev_row 对应 DP[i-1][...]
        # 初始化为与空串 s1 对齐的代价: 0, delta, 2*delta ...
        prev_row = [j * self.DELTA for j in range(n + 1)]
        curr_row = [0] * (n + 1)

        for i in range(1, m + 1):
            # curr_row[0] 对应 DP[i][0] = i * delta
            curr_row[0] = i * self.DELTA
            for j in range(1, n + 1):
                char1 = s1[i - 1]
                char2 = s2[j - 1]

                cost_match = prev_row[j - 1] + self.ALPHA[char1][char2]
                cost_gap1 = prev_row[j] + self.DELTA  # s1 gap (上方)
                cost_gap2 = curr_row[j - 1] + self.DELTA  # s2 gap (左方)

                curr_row[j] = min(cost_match, cost_gap1, cost_gap2)

            # 滚动数组：当前行变成下一轮的“上一行”
            prev_row = list(curr_row)

        return prev_row

    def solve_basic(self, s1, s2):
        """
        基础版 DP 算法：用于解决分治后的 Base Case (小规模问题)。
        当 m <= 2 时，直接用这个解，避免过度递归。
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 初始化边界
        for i in range(1, m + 1):
            dp[i][0] = i * self.DELTA
        for j in range(1, n + 1):
            dp[0][j] = j * self.DELTA

        # 填表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                char1 = s1[i - 1]
                char2 = s2[j - 1]
                dp[i][j] = min(
                    dp[i - 1][j - 1] + self.ALPHA[char1][char2],
                    dp[i - 1][j] + self.DELTA,
                    dp[i][j - 1] + self.DELTA
                )

        # 回溯找对齐串
        align1, align2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            char1 = s1[i - 1] if i > 0 else None
            char2 = s2[j - 1] if j > 0 else None
            curr = dp[i][j]

            if i > 0 and j > 0 and curr == dp[i - 1][j - 1] + self.ALPHA[char1][char2]:
                align1.append(char1)
                align2.append(char2)
                i -= 1;
                j -= 1
            elif i > 0 and curr == dp[i - 1][j] + self.DELTA:
                align1.append(char1)
                align2.append('_')
                i -= 1
            else:
                align1.append('_')
                align2.append(char2)
                j -= 1

        return dp[m][n], "".join(align1[::-1]), "".join(align2[::-1])

    def divide_and_conquer(self, s1, s2):
        """
        Hirschberg 算法核心递归逻辑
        """
        m = len(s1)
        n = len(s2)

        # Base case: 当 s1 长度很小时（例如 <= 2），不再分割，直接求解
        if m <= 2:
            return self.solve_basic(s1, s2)

        # 1. 分割: 将 s1 从中间切开
        mid = m // 2
        s1_left = s1[:mid]
        s1_right = s1[mid:]

        # 2. 计算分割线上的得分
        # Score L: s1的前半部分 vs s2的所有前缀 (常规 DP)
        score_l = self.get_last_row_score(s1_left, s2)

        # Score R: s1的后半部分(反转) vs s2的所有后缀(反转)
        # 这里通过反转字符串来复用 get_last_row_score 逻辑
        score_r = self.get_last_row_score(s1_right[::-1], s2[::-1])

        # 3. 寻找最佳分割点 split_idx (即 k)
        # 我们需要找到一个 k，使得 score_l[k] + score_r[n-k] 最小
        # 注意: score_r 是基于反转 s2 计算的，所以 score_r 的索引 i 对应原 s2 的倒数第 i 个位置
        min_cost = float('inf')
        split_idx = -1

        for k in range(n + 1):
            # 左边匹配 s2[:k] 的代价 + 右边匹配 s2[k:] 的代价
            total = score_l[k] + score_r[n - k]
            if total < min_cost:
                min_cost = total
                split_idx = k

        # 4. 递归求解左右两半
        # 左子问题: s1前半段 和 s2的前 split_idx 个字符
        cost_l, align1_l, align2_l = self.divide_and_conquer(s1_left, s2[:split_idx])

        # 右子问题: s1后半段 和 s2剩下的部分
        cost_r, align1_r, align2_r = self.divide_and_conquer(s1_right, s2[split_idx:])

        # 5. 合并结果
        return cost_l + cost_r, align1_l + align1_r, align2_l + align2_r

    def solve(self, s1, s2):
        return self.divide_and_conquer(s1, s2)


def process_memory():
    """
    获取当前进程的内存占用 (单位 KB)
    """
    # 优先使用 resource (Linux/提交环境标准)
    if resource:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage  # Linux 通常返回 KB

    # Windows 开发环境兼容
    elif psutil:
        process = psutil.Process()
        memory_info = process.memory_info()
        return int(memory_info.rss / 1024)  # 转换为 KB

    else:
        return 0


def measure_performance():
    # 处理命令行参数
    if len(sys.argv) < 3:
        input_path = 'input.txt'
        output_path = 'output.txt'
        print(f"【提示】未检测到命令行参数，使用默认文件：输入={input_path}, 输出={output_path}")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]

    # 1. 生成字符串
    generator = InputGenerator(input_path)
    s1, s2 = generator.generate()

    solver = EfficientSolver()

    # 2. 开始计时
    start_time = time.time()

    # 3. 运行算法
    cost, align1, align2 = solver.solve(s1, s2)

    # 4. 结束计时
    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000

    # 5. 获取内存
    memory_kb = process_memory()

    # 6. 写入文件
    with open(output_path, 'w') as f:
        f.write(f"{cost}\n")
        f.write(f"{align1}\n")
        f.write(f"{align2}\n")
        f.write(f"{time_taken_ms}\n")
        f.write(f"{memory_kb}\n")


if __name__ == "__main__":
    measure_performance()