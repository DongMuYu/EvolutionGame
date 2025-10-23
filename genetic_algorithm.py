import random
import numpy as np
import os
import setting
from neural_input_config import get_input_size

class GeneticAlgorithm:
    """
    遗传算法类，用于优化神经网络的权重参数。
    神经网络结构：input_size个输入 -> 128个ReLU神经元 -> 64个ReLU神经元 -> 32个ReLU神经元 -> 16个ReLU神经元 -> 2个输出（上下、左右）
    """

    def __init__(self, population_size, input_size=None, 
        hidden1_size=setting.NEURAL_NETWORK_HIDDEN1_SIZE, 
        hidden2_size=setting.NEURAL_NETWORK_HIDDEN2_SIZE, 
        hidden3_size=setting.NEURAL_NETWORK_HIDDEN3_SIZE, 
        hidden4_size=setting.NEURAL_NETWORK_HIDDEN4_SIZE, 
        output_size=setting.NEURAL_NETWORK_OUTPUT_SIZE):
        self.population_size = population_size
        
        # 神经网络结构参数
        self.input_size = input_size if input_size is not None else get_input_size()
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        self.output_size  = output_size
        
        # 计算染色体长度（与个体类保持一致）
        # 输入层到隐藏层1的权重 + 隐藏层1偏置
        # 隐藏层1到隐藏层2的权重 + 隐藏层2偏置
        # 隐藏层2到隐藏层3的权重 + 隐藏层3偏置
        # 隐藏层3到隐藏层4的权重 + 隐藏层4偏置
        # 隐藏层4到输出层的权重 + 输出层偏置
        self.chromosome_length = (self.input_size * self.hidden1_size + self.hidden1_size + 
                                 self.hidden1_size * self.hidden2_size + self.hidden2_size + 
                                 self.hidden2_size * self.hidden3_size + self.hidden3_size + 
                                 self.hidden3_size * self.hidden4_size + self.hidden4_size + 
                                 self.hidden4_size * self.output_size + self.output_size)
        
        # 创建保存目录
        self.save_dir = "saved_models"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def get_network_config(self):
        """
        获取神经网络配置
        
        返回:
            dict: 包含神经网络结构参数的字典
        """
        return {
            'input_size': self.input_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'hidden3_size': self.hidden3_size,
            'hidden4_size': self.hidden4_size,
            'output_size': self.output_size,
            'chromosome_length': self.chromosome_length
        }

    def _initialize_population(self):
        """
        初始化种群 - 使用更小的随机权重，并为每个个体添加变异率和变异大小属性。

        返回:
            list: 包含初始化权重的种群列表，每个个体是一个字典，包含weights、mutation_rates和mutation_magnitudes。
        """
        # 初始化种群列表
        population = []
        for _ in range(self.population_size):
            # 使用较小的初始权重，避免极端行为
            weights = np.random.randn(self.chromosome_length) * 0.5
            
            # 使用权重数组作为个体
            population.append(weights)
        return population

    def select(self, population, fitness_scores):
        """
         
        当前弃用轮盘赌，使用微生物遗传算法
        选择操作 - 轮盘赌选择，加入精英保留。

        参数:
            population (list): 当前种群中的个体列表。
            fitness_scores (numpy.ndarray): 每个个体对应的适应度得分。

        返回:
            list: 经过选择后的新种群，包含除一个位置外的所有个体以及一个精英个体。
        """
        # 防止除零
        # 如果总适应度为0
        if np.sum(fitness_scores) == 0:
            # 个体被选中概率相等
            probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
        else:
            # 加入最小概率，避免完全随机
            min_prob = 0.0001 / len(fitness_scores)

            # 每个个体的概率等于其适应度得分除以总适应度得分
            probabilities = fitness_scores / np.sum(fitness_scores)

            # 确保每个个体的概率至少不低于之前设定的最小概率min_pro
            probabilities = np.maximum(probabilities, min_prob)

            # 前面的操作可能会导致概率总和不等于1，这行代码重新对所有概率进行归一化处理，确保它们的总和为1
            probabilities = probabilities / np.sum(probabilities)  # 重新归一化

        # 根据选择概率随机选择个体索引，选择数量为种群大小减1（留一个位置给精英个体）
        selected_indices = np.random.choice(
            len(population),                      # 选择范围：种群中的所有个体索引(0到种群大小-1)
            size=len(population) - 1,             # 选择的个体数量：种群大小减1
            p=probabilities,                      # 每个个体被选中的概率
            replace=True                          # 是否有放回抽样（允许同一个体被多次选中）
        )

        # 精英选择：找到适应度最高的个体索引
        elite_index = np.argmax(fitness_scores)

        # 根据选择的索引构建新种群
        selected_population = [population[i] for i in selected_indices]

        # 将精英个体（最佳个体）复制添加到新种群末尾，确保优秀基因不会丢失
        selected_population.append(population[elite_index].copy())  # 保留精英

        return selected_population

    def _calculate_population_diversity(self, population):
        """
        当前弃用
        计算种群多样性
        
        通过计算种群中所有个体之间的平均欧氏距离来衡量多样性。
        多样性指标被归一化到0-1范围，其中0表示种群完全一致，1表示最大多样性。
        
        参数:
            population (list): 种群列表，每个个体是包含权重、变异率和变异大小的字典
            
        返回:
            float: 多样性指标，范围在0-1之间
        """
        # 特殊情况处理：单个个体或空种群
        if len(population) <= 1:
            return 1.0  # 单个个体时多样性最高（因为没有其他个体可以比较）
            
        # 计算所有个体之间的平均欧氏距离
        total_distance = 0
        count = 0
        
        # 遍历所有个体对（避免重复计算）
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # 计算两个个体权重向量之间的欧氏距离
                # 欧氏距离公式：sqrt(Σ(x_i - y_i)^2)
                distance = np.linalg.norm(population[i]['weights'] - population[j]['weights'])
                total_distance += distance
                count += 1
        
        # 防止除零错误
        if count == 0:
            return 0.0
            
        # 计算平均距离
        avg_distance = total_distance / count
        
        # 归一化到0-1范围
        # 最大预期距离基于染色体长度和标准差的经验值
        # 假设权重在[-1,1]范围内，最大可能距离是sqrt(n)*2
        max_expected_distance = np.sqrt(self.chromosome_length) * 2.0  # 经验阈值
        
        # 确保多样性指标在0-1范围内
        diversity = min(avg_distance / max_expected_distance, 1.0)
        
        return diversity
        
    def crossover(self, parent1, parent2, crossover_type="blend"):

        """
        当前弃用
        复杂高效的交叉操作 - 支持多种交叉策略。

        参数:
            parent1 (numpy.ndarray): 第一个父代个体。
            parent2 (numpy.ndarray): 第二个父代个体。
            crossover_type (str): 交叉类型，可选 "blend"(混合交叉), "simulated_binary"(模拟二进制交叉), 
                                "arithmetic"(算术交叉), "uniform"(均匀交叉)

        返回:
            tuple: 两个子代个体 (child1, child2)，均为 numpy 数组。
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if crossover_type == "uniform":
            # 均匀交叉：每个基因有50%概率来自父代1或父代2
            for i in range(len(parent1)):
                if random.random() > 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
                    
        elif crossover_type == "blend":
            # BLX-α混合交叉：在父代周围扩展搜索空间
            alpha = 0.5  # 扩展因子
            for i in range(len(parent1)):
                # 计算父代基因的最小值和最大值
                min_gene = min(parent1[i], parent2[i])
                max_gene = max(parent1[i], parent2[i])
                
                # 计算扩展范围
                range_gene = max_gene - min_gene
                lower_bound = min_gene - alpha * range_gene
                upper_bound = max_gene + alpha * range_gene
                
                # 在扩展范围内随机生成子代基因
                child1[i] = random.uniform(lower_bound, upper_bound)
                child2[i] = random.uniform(lower_bound, upper_bound)
                
        elif crossover_type == "simulated_binary":
            # 模拟二进制交叉(SBX)：模拟二进制交叉的连续版本
            eta_c = 20  # 分布指数
            for i in range(len(parent1)):
                if random.random() <= 0.9:  # 交叉概率
                    u = random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                else:
                    # 不交叉，直接复制父代
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
                    
        elif crossover_type == "arithmetic":
            # 算术交叉：线性组合
            alpha = random.random()  # 随机权重
            for i in range(len(parent1)):
                child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                child2[i] = (1 - alpha) * parent1[i] + alpha * parent2[i]
        
        return child1, child2
    
    def _select_crossover_strategy(self, diversity, fitness_range):
        """
        当前弃用
        根据种群多样性和适应度范围选择交叉策略
        
        参数:
            diversity (float): 种群多样性指标
            fitness_range (float): 适应度范围
            
        返回:
            str: 交叉策略类型
        """
        if diversity < 0.2:
            # 种群多样性低时使用混合交叉，扩大搜索空间
            return "blend"
        elif fitness_range < 100:  # 适应度差异较小时
            # 使用模拟二进制交叉进行精细搜索
            return "simulated_binary"
        else:
            # 默认使用算术交叉
            return "arithmetic"

    def mutate_gradient(self, individual, rank_percentage=None):
        """
        梯度变异操作 - 根据个体排名应用不同的变异强度和概率。

        参数:
            individual (numpy.ndarray): 待变异的个体权重数组。
            rank_percentage (float): 个体在种群中的排名百分比（0-1），越小表示排名越靠前。

        返回:
            numpy.ndarray: 经过变异处理后的个体权重数组。
        """
        mutated_individual = individual.copy()
        
        # 如果没有提供排名百分比, 抛出异常
        if rank_percentage is None:
            raise ValueError("必须提供个体排名百分比（0-1）")
        
        # 梯度变异策略：适应度越低，变异强度和变异概率越高
        # 变异概率：从20%到50%梯度上升
        mutation_rate = 0.2 + (rank_percentage * 0.3)  # 20% - 50%
        
        # 变异强度：从0.1到0.5梯度上升  
        mutation_strength = 0.1 + (rank_percentage * 0.4)  # 0.1 - 0.5
        
        # 对每个基因进行变异
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                # 使用梯度变异强度进行变异
                mutation = np.random.randn() * mutation_strength
                mutated_individual[i] += mutation
        
        return mutated_individual

    def mutate(self, individual, mutation_rate=0.3, mutation_strength=0.2):
        """
        普通变异操作 - 微生物遗传算法的标准变异。

        参数:
            individual (numpy.ndarray): 待变异的个体权重数组。
            mutation_rate (float): 变异概率，默认30%。
            mutation_strength (float): 变异强度，默认0.2。

        返回:
            numpy.ndarray: 经过变异处理后的个体权重数组。
        """
        mutated_individual = individual.copy()
        
        # 对每个基因进行变异
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                # 使用标准变异强度进行变异
                mutation = np.random.randn() * mutation_strength
                mutated_individual[i] += mutation
        
        return mutated_individual

    def evolve(self, population, fitness_scores):
        """
        执行一代进化 
        - 新策略：前10%优秀个体完全保留，10%-50%个体随机选择变异+接收优秀基因片段
        - 50%-100% 直接随机复制前10%的个体并采取梯度增长的变异策略
        - 梯度变异策略：适应度越低，变异强度和变异概率越高
        - 变异强度：从0.1到0.5梯度上升
        - 变异概率：从20%到50%梯度上升 
        
        参数:
            population (list): 当前种群中的个体列表，每个个体是包含weights、mutation_rates和mutation_magnitudes的字典。
            fitness_scores (numpy.ndarray): 每个个体对应的适应度得分。

        返回:
            list: 新一代的种群列表。
        """
        # 验证输入参数长度一致性
        actual_population_size = len(population)
        actual_fitness_size = len(fitness_scores)
        
        if actual_population_size != actual_fitness_size:
            raise ValueError(f"种群大小({actual_population_size})与适应度分数数量({actual_fitness_size})不一致")
        else:
            effective_size = actual_population_size
        
        # 根据适应度排序，获取排序索引
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降序排序
        
        # 计算各分组的边界，基于有效大小
        top_10_count = max(1, int(effective_size * 0.10))  # 前10%优秀个体
        middle_40_count = max(0, int(effective_size * 0.40))  # 10%-50%的个体
        bottom_50_count = effective_size - top_10_count - middle_40_count  # 50%-100%的个体
        
        # 获取各分组个体索引
        top_10_indices = sorted_indices[:top_10_count]
        middle_40_indices = sorted_indices[top_10_count:top_10_count + middle_40_count]
        bottom_50_indices = sorted_indices[top_10_count + middle_40_count:]
        
        # 创建新一代种群
        new_population = []
        
        # 1. 前10%优秀个体完全保留
        for idx in top_10_indices:
            new_population.append(population[idx].copy())
        
        # 2. 10%-50%个体：先执行普通变异，再接收随机长度的优秀基因片段
        for idx in middle_40_indices:
            # 先执行普通微生物遗传算法变异（30%概率，强度0.2）
            mutated_individual = self.mutate(population[idx], mutation_rate=0.3, mutation_strength=0.2)
            
            # 接收优秀基因片段：从前10%优秀个体中随机选择一个，并复制随机长度的基因
            elite_parent = population[random.choice(top_10_indices)]
            
            # 随机选择基因片段长度（10%-30%的基因）
            min_segment_size = max(1, int(len(mutated_individual) * 0.1))
            max_segment_size = max(1, int(len(mutated_individual) * 0.3))
            gene_segment_size = random.randint(min_segment_size, max_segment_size)
            
            # 确保片段大小不超过个体长度
            gene_segment_size = min(gene_segment_size, len(mutated_individual))
            
            # 随机选择起始位置
            start_pos = random.randint(0, len(mutated_individual) - gene_segment_size)
            
            # 复制优秀基因片段，替代原有同位置的基因
            mutated_individual[start_pos:start_pos + gene_segment_size] = elite_parent[start_pos:start_pos + gene_segment_size]
            
            new_population.append(mutated_individual)
        
        # 3. 50%-100%个体：随机复制前10%个体并采取梯度增长的变异策略
        for idx in bottom_50_indices:
            # 随机复制前10%优秀个体
            elite_parent = population[random.choice(top_10_indices)]
            
            # 计算个体排名百分比（0-1，越小表示排名越靠前）
            rank_percentage = (sorted_indices.tolist().index(idx) + 1) / effective_size
            
            # 应用梯度变异
            mutated_individual = self.mutate_gradient(elite_parent, rank_percentage)
            new_population.append(mutated_individual)
        
        # 确保新种群大小正确
        assert len(new_population) == self.population_size, f"新种群大小错误: {len(new_population)} != {self.population_size}"
        
        return new_population
        
    def save_population(self, population, generation, fitness_scores):
        """
        保存种群数据到文件（微生物遗传算法简化版）
        
        参数:
            population (list): 当前种群，每个个体是权重数组。
            generation (int): 当前代数。
            fitness_scores (list): 适应度分数列表。
        """
        # 创建保存文件名，最佳适应度值保留小数点后3位
        best_fitness = max(fitness_scores) if len(fitness_scores) > 0 else 0
        avg_fitness = np.mean(fitness_scores) if len(fitness_scores) > 0 else 0
        best_fitness_str = f"{best_fitness:.3f}"
        avg_fitness_str = f"{avg_fitness:.3f}"
        
        # 保存种群数据
        population_filename = f"{self.save_dir}/generation_{generation}_fitness_{best_fitness_str}_avg_{avg_fitness_str}.npz"
        
        # 提取权重数据
        weights_array = np.array(population)
        fitness_scores_array = np.array(fitness_scores)
        
        # 保存种群数据
        np.savez_compressed(population_filename, 
                           weights=weights_array,
                           generation=generation,
                           fitness_scores=fitness_scores_array,
                           best_fitness=best_fitness,
                           avg_fitness=avg_fitness,
                           chromosome_length=self.chromosome_length,
                           population_size=self.population_size,
                           model_structure={
                               'input_size': self.input_size,
                               'hidden1_size': self.hidden1_size,
                               'hidden2_size': self.hidden2_size,
                               'hidden3_size': self.hidden3_size,
                               'hidden4_size': self.hidden4_size,
                               'output_size': self.output_size
                           })
        
        print(f"种群数据已保存至: {population_filename}")
        
    def load_population(self, filepath):
        """
        从文件加载种群数据（微生物遗传算法简化版）
        支持加载旧模型并处理神经网络结构不匹配的情况
        
        参数:
            filepath (str): 保存的文件路径。
            
        返回:
            tuple: (population, generation, fitness_scores, best_fitness, avg_fitness) 加载的种群数据和相关信息。
        """
        # 加载数据，需要设置allow_pickle=True来加载旧格式的numpy数组
        data = np.load(filepath, allow_pickle=True)
        generation = int(data['generation'])
        
        # 获取适应度分数
        fitness_scores = data['fitness_scores'].tolist()
        
        # 获取最佳适应度和平均适应度
        best_fitness = float(data['best_fitness']) if 'best_fitness' in data else max(fitness_scores) if fitness_scores else 0
        avg_fitness = float(data['avg_fitness']) if 'avg_fitness' in data else np.mean(fitness_scores) if fitness_scores else 0
        
        # 检查旧模型的神经网络结构
        old_model_structure = None
        if 'model_structure' in data:
            old_model_structure = data['model_structure'].item()
            print(f"加载的模型结构: {old_model_structure}")
        
        # 加载权重数据
        weights_array = data['weights']
        
        # 处理权重数据，转换为新的种群格式
        population = []
        
        for i in range(len(weights_array)):
            old_weights = weights_array[i]
            
            # 检查神经网络结构是否匹配
            structure_matches = True
            if old_model_structure:
                # 检查各层大小是否匹配
                if (old_model_structure.get('input_size', self.input_size) != self.input_size or
                    old_model_structure.get('hidden1_size', self.hidden1_size) != self.hidden1_size or
                    old_model_structure.get('hidden2_size', self.hidden2_size) != self.hidden2_size or
                    old_model_structure.get('hidden3_size', self.hidden3_size) != self.hidden3_size or
                    old_model_structure.get('hidden4_size', self.hidden4_size) != self.hidden4_size or
                    old_model_structure.get('output_size', self.output_size) != self.output_size):
                    structure_matches = False
            
            if structure_matches and len(old_weights) == self.chromosome_length:
                # 如果结构匹配且权重大小正确，直接使用原始权重
                new_weights = old_weights
            else:
                # 如果结构不匹配，重新初始化权重
                print(f"神经网络结构不匹配，为个体{i}重新初始化权重")
                print(f"旧权重大小: {len(old_weights)}, 新权重大小: {self.chromosome_length}")
                
                # 使用较小的初始权重，避免极端行为
                new_weights = np.random.randn(self.chromosome_length) * 0.5
                
                # 如果旧模型是3输出，新模型是2输出，尝试部分转换
                if old_model_structure and old_model_structure.get('output_size', 3) == 3 and self.output_size == 2:
                    try:
                        # 计算旧模型各层权重的大小
                        old_input_to_h1_size = old_model_structure.get('input_size', self.input_size) * old_model_structure.get('hidden1_size', self.hidden1_size)
                        old_h1_bias_size = old_model_structure.get('hidden1_size', self.hidden1_size)
                        old_h1_to_h2_size = old_model_structure.get('hidden1_size', self.hidden1_size) * old_model_structure.get('hidden2_size', self.hidden2_size)
                        old_h2_bias_size = old_model_structure.get('hidden2_size', self.hidden2_size)
                        old_h2_to_output_size = old_model_structure.get('hidden2_size', self.hidden2_size) * old_model_structure.get('output_size', 3)
                        old_output_bias_size = old_model_structure.get('output_size', 3)
                        
                        # 从旧权重中提取前两层（如果可能）
                        if len(old_weights) >= old_input_to_h1_size + old_h1_bias_size + old_h1_to_h2_size + old_h2_bias_size:
                            w1_old = old_weights[0:old_input_to_h1_size].reshape(old_model_structure.get('input_size', self.input_size), old_model_structure.get('hidden1_size', self.hidden1_size))
                            b1_old = old_weights[old_input_to_h1_size:old_input_to_h1_size+old_h1_bias_size]
                            w2_old = old_weights[old_input_to_h1_size+old_h1_bias_size:old_input_to_h1_size+old_h1_bias_size+old_h1_to_h2_size].reshape(old_model_structure.get('hidden1_size', self.hidden1_size), old_model_structure.get('hidden2_size', self.hidden2_size))
                            b2_old = old_weights[old_input_to_h1_size+old_h1_bias_size+old_h1_to_h2_size:old_input_to_h1_size+old_h1_bias_size+old_h1_to_h2_size+old_h2_bias_size]
                            
                            # 如果旧模型的前两层与新模型的前两层大小匹配，则使用旧权重
                            if (old_model_structure.get('input_size', self.input_size) == self.input_size and
                                old_model_structure.get('hidden1_size', self.hidden1_size) == self.hidden1_size and
                                old_model_structure.get('hidden2_size', self.hidden2_size) == self.hidden2_size):
                                
                                # 将前两层权重复制到新权重中
                                new_weights[0:old_input_to_h1_size] = w1_old.flatten()
                                new_weights[old_input_to_h1_size:old_input_to_h1_size+old_h1_bias_size] = b1_old
                                new_weights[old_input_to_h1_size+old_h1_bias_size:old_input_to_h1_size+old_h1_bias_size+old_h1_to_h2_size] = w2_old.flatten()
                                new_weights[old_input_to_h1_size+old_h1_bias_size+old_h1_to_h2_size:old_input_to_h1_size+old_h1_bias_size+old_h1_to_h2_size+old_h2_bias_size] = b2_old
                                
                                print(f"个体{i}的前两层权重已从旧模型转换")
                    except Exception as e:
                        print(f"权重转换失败: {e}")
            
            # 直接使用权重数组作为个体
            population.append(new_weights)
        
        print(f"种群数据已从 {filepath} 加载，共{len(population)}个个体")
        return population, generation, fitness_scores, best_fitness, avg_fitness