# setting.py

# ===== 显示设置 =====
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60
CAPTION = "Genetic Algorithm"

# ===== 颜色定义 (R, G, B) =====
WHITE = (255, 255, 255)    # 白色
YELLOW = (255, 255, 0)     # 黄色
BLACK = (0, 0, 0)          # 黑色
RED = (255, 0, 0)          # 红色
GREEN = (0, 255, 0)        # 绿色
BLUE = (0, 120, 255)       # 蓝色
DARK_BLUE = (0, 0, 139)    # 深蓝色
SKY_BLUE = (135, 206, 235) # 天蓝色
PURPLE = (128, 0, 128)     # 紫色
ORANGE = (255, 165, 0)     # 橙色

# ===== 个体参数 =====
INDIVIDUAL_WIDTH = 30   # 个体宽度
INDIVIDUAL_HEIGHT = 30  # 个体高度
INDIVIDUAL_RADIUS = 15  # 个体半径（圆形个体）
INDIVIDUAL_HORIZONTAL_SPEED = 6  # 个体水平移动速度
INDIVIDUAL_VERTICAL_SPEED = 6  # 个体垂直移动速度
INDIVIDUAL_MAX_FLIGHT_ENERGY = 100  # 个体飞行最大能量值
FLIGHT_ENERGY_COST_PERCENT = 0.03  # 每次飞行扣除的能量百分比
FLIGHT_ENERGY_RECOVERY_PERCENT = 0.10  # 每帧恢复的能量百分比
GRAVITY_ACCELERATION = 0.8  # 重力加速度，每帧向下的加速度
MAX_RAY_DISTANCE = 250.0  # 最大射线检测距离

# ===== 目标参数 =====
GOAL_SIZE = 15  # 目标大小（半径）
GOAL_COUNT = 5  # 目标数量

# ===== 游戏帧计时参数 =====
ROUND_DURATION_FRAMES = 10000  # 每轮持续帧数

# ===== 遗传算法参数 =====
POPULATION_SIZE = 100  # 种群个体数
SAVE_EVERY_N_GENERATIONS = 100  # 每N代保存一次数据

# ===== 神经网络参数 =====
NEURAL_NETWORK_HIDDEN1_SIZE = 128  # 第一隐藏层神经元数量
NEURAL_NETWORK_HIDDEN2_SIZE = 64   # 第二隐藏层神经元数量
NEURAL_NETWORK_HIDDEN3_SIZE = 32   # 第三隐藏层神经元数量
NEURAL_NETWORK_HIDDEN4_SIZE = 16   # 第四隐藏层神经元数量
NEURAL_NETWORK_OUTPUT_SIZE = 2     # 输出层神经元数量



