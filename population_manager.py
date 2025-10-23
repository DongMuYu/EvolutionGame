import numpy as np
import os
import re
from entities.individual import Individual
import setting
from genetic_algorithm import GeneticAlgorithm

class PopulationManager:
    """
    种群管理类，负责创建种群、计算适应度和执行进化
    """
    
    def __init__(self):
        # 遗传算法 - 敌人种群
        self.genetic_algorithm = GeneticAlgorithm(population_size=setting.POPULATION_SIZE)
        
        # 模型保存目录
        self.save_dir = "saved_models"
        
        # 当前代数
        self.current_generation = 1
        
        # 尝试加载最新模型，如果失败则初始化新种群
        self.current_population = self._load_latest_model()
        if self.current_population is None:
            self.current_population = self.genetic_algorithm._initialize_population()
            print("未找到保存的模型，初始化新种群")
        else:
            print("成功加载最新模型，继续训练")
    
    def _load_latest_model(self):
        """加载最新的模型文件"""
        # 检查模型目录是否存在
        if not os.path.exists(self.save_dir):
            print("模型目录不存在，将创建新种群")
            return None
            
        # 获取所有模型文件
        model_files = [f for f in os.listdir(self.save_dir) if f.endswith('.npz')]
        
        if not model_files:
            print("未找到模型文件，将创建新种群")
            return None
            
        # 从文件名中提取代数
        def extract_generation(filename):
            match = re.search(r'generation_(\d+)_fitness', filename)
            if match:
                return int(match.group(1))
            return 0
        
        # 按代数排序，选择最新的模型
        latest_model = max(model_files, key=extract_generation)
        
        try:
            # 构造完整的文件路径
            model_path = os.path.join(self.save_dir, latest_model)
            
            # 加载模型
            population_list, generation, fitness_scores, best_fitness, avg_fitness = self.genetic_algorithm.load_population(model_path)
            print(f"加载模型: {latest_model}, 代数: {generation}, 最佳适应度: {best_fitness:.3f}, 平均适应度: {avg_fitness:.3f}")
            
            # 保存加载的代数信息
            self.current_generation = generation
            return population_list
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈
            return None
    
    def get_current_generation(self):
        """获取当前代数"""
        return self.current_generation
    
    def initialize_individuals(self, pos_x, pos_y, frame_counter_ref=None):
        """初始化种群个体
        
        参数:
            pos_x: 个体初始 x 坐标
            pos_y: 个体初始 y 坐标
            frame_counter_ref: 帧计数器引用函数
        """
        individuals = []
        
        # 生成固定数量的种群个体，与种群大小一致
        individual_count = setting.POPULATION_SIZE  # 种群个体数量
        
        for i in range(individual_count):
            
            # 为每个种群个体分配对应的遗传算法个体
            if i < len(self.current_population):
                # 使用种群个体
                neural_network = self.current_population[i]
            else:
                neural_network = None
                
            # 获取遗传算法的神经网络配置
            network_config = self.genetic_algorithm.get_network_config()
            
            individual = Individual(
                                  width=setting.INDIVIDUAL_WIDTH,
                                  height=setting.INDIVIDUAL_HEIGHT,
                                  pos_x=pos_x, pos_y=pos_y, individual_id=i, 
                                  neural_network=neural_network, frame_counter_ref=frame_counter_ref,
                                  input_size=network_config['input_size'],
                                  hidden1_size=network_config['hidden1_size'],
                                  hidden2_size=network_config['hidden2_size'],
                                  hidden3_size=network_config['hidden3_size'],
                                  hidden4_size=network_config['hidden4_size'],
                                  output_size=network_config['output_size'],
)
            individuals.append(individual)
        
        return individuals
    
    def update_fitness(self, individuals, goals, obstacles):
        """更新种群个体适应度
        
        参数:
            individuals: 个体列表
            goals: 目标列表
            obstacles: 障碍物列表
        """
        individual_fitness = {}
        
        # 统计每个种群个体的适应度，强制重新计算适应度
        for individual in individuals:
            # 强制重新计算适应度，不使用存储的值
            fitness = individual.calculate_fitness(goals, obstacles)
            individual.fitness = fitness
                
            individual_fitness[individual.entity_id] = fitness
            
        return individual_fitness
    
    def evolve_population(self, individuals, individual_fitness, generation, goals=None):
        """执行遗传算法进化"""
        # 收集种群个体的适应度
        fitness_scores = []
        for i in range(setting.POPULATION_SIZE):  # 所有种群个体（与种群大小一致）
            if i in individual_fitness:
                fitness_scores.append(individual_fitness[i])
            else:
                raise ValueError(f"个体 {i} 没有适应度值")
        
        # 转换为numpy数组
        fitness_scores = np.array(fitness_scores)
        
        # 显示适应度信息
        max_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        min_fitness = np.min(fitness_scores)
        
        # 计算平均到达帧数
        avg_reach_frames = []
        for individual in individuals:
            if individual.goal_reach_times:
                avg_reach_frames.append(sum(individual.goal_reach_times) / len(individual.goal_reach_times))
        
        min_avg_frames = min(avg_reach_frames) if avg_reach_frames else 0
        max_goals_reached = max([individual.goals_reached for individual in individuals], default=0)
        successful_individuals = len([ind for ind in individuals if ind.goal_reach_times])
        alive_individuals = len([ind for ind in individuals if ind.active])
        
        # 找出最佳、中位数和最差个体
        best_idx = np.argmax(fitness_scores)
        worst_idx = np.argmin(fitness_scores)
        median_idx = np.argsort(fitness_scores)[len(fitness_scores)//2]
        
        # 获取这些个体的详细信息
        best_individual = individuals[best_idx]
        worst_individual = individuals[worst_idx]
        median_individual = individuals[median_idx]
        
        print(f"第{generation}代进化完成:")
        print(f"  最高适应度: {max_fitness:.3f}")
        print(f"  平均适应度: {avg_fitness:.3f}")
        print(f"  最低适应度: {min_fitness:.3f}")
        print(f"  最短平均到达帧数: {min_avg_frames:.1f}帧")
        print(f"  最多目标达成数: {max_goals_reached}")
        
        # 打印最佳个体信息
        print("\n=== 最佳个体信息 ===")
        print(f"  个体ID: {best_individual.entity_id}")
        print(f"  适应度: {fitness_scores[best_idx]:.3f}")
        print(f"  到达目标数: {best_individual.goals_reached}")
        print(f"  存活状态: {'存活' if best_individual.active else '死亡'}")
        if best_individual.goal_reach_times:
            print(f"  平均到达帧数: {sum(best_individual.goal_reach_times) / len(best_individual.goal_reach_times):.1f}")
        else:
            print(f"  平均到达帧数: 无到达记录")
        
        # 打印中位数成绩个体信息
        print("\n=== 中位数成绩个体信息 ===")
        print(f"  个体ID: {median_individual.entity_id}")
        print(f"  适应度: {fitness_scores[median_idx]:.3f}")
        print(f"  到达目标数: {median_individual.goals_reached}")
        print(f"  存活状态: {'存活' if median_individual.active else '死亡'}")
        if median_individual.goal_reach_times:
            print(f"  平均到达帧数: {sum(median_individual.goal_reach_times) / len(median_individual.goal_reach_times):.1f}")
        else:
            print(f"  平均到达帧数: 无到达记录")
        
        # 打印最差个体信息
        print("\n=== 最差个体信息 ===")
        print(f"  个体ID: {worst_individual.entity_id}")
        print(f"  适应度: {fitness_scores[worst_idx]:.3f}")
        print(f"  到达目标数: {worst_individual.goals_reached}")
        print(f"  存活状态: {'存活' if worst_individual.active else '死亡'}")
        if worst_individual.goal_reach_times:
            print(f"  平均到达帧数: {sum(worst_individual.goal_reach_times) / len(worst_individual.goal_reach_times):.1f}")
        else:
            print(f"  平均到达帧数: 无到达记录")
        
        # 执行进化
        self.current_population = self.genetic_algorithm.evolve(
            self.current_population, fitness_scores
        )
        
        # 更新当前代数
        self.current_generation = generation + 1
        
        # 只在每N代或关闭游戏时保存模型和数据
        if generation % setting.SAVE_EVERY_N_GENERATIONS == 0:
            # 保存种群模型
            self.genetic_algorithm.save_population(
                self.current_population, generation, fitness_scores
            )
        
        print(f"\n种群已进化到第{generation + 1}代")
        
        # 打印最后适应度统计信息
        print("\n=== 最后适应度统计 ===")
        print(f"第{generation}代最终统计:")
        print(f"  最高适应度: {max_fitness:.3f}")
        print(f"  平均适应度: {avg_fitness:.3f}")
        print(f"  最低适应度: {min_fitness:.3f}")
        print(f"  最短平均到达帧数: {min_avg_frames:.1f}帧")
        print(f"  最多目标达成数: {max_goals_reached}")
        print(f"  成功到达目标的个体数: {successful_individuals}")
        print(f"  存活个体数: {alive_individuals}")
        print("====================")
        
        return max_fitness, avg_fitness, min_avg_frames, max_goals_reached