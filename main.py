import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 参数配置
NUM_NODES = 5  # 总节点数（包含1个根节点）
HIDDEN_DIM = 128  # 神经网络隐藏层维度
LEARNING_RATE = 0.001  # 学习率
NUM_EPISODES = 1000  # 训练轮数

# 参数配置（新增或修改）
TRANSFORMER_NHEADS = 2          # 注意力头数（需能被d_model整除）
TRANSFORMER_DIM_FF = 256        # 前馈网络隐藏层维度
TRANSFORMER_LAYERS = 2          # Transformer层数
DROPOUT = 0.1                   # 防止过拟合


class NetworkEnvironment:
    """模拟网络环境生成器"""

    def __init__(self):
        # 生成节点位置（0号节点为根节点）
        # 生成NUM_NODES数目的随机坐标
        self.node_pos = np.random.rand(NUM_NODES, 2) * 100

        # 计算节点间距离矩阵
        # 建立一个全零矩阵
        self.dist_matrix = np.zeros((NUM_NODES, NUM_NODES))
        # 计算节点i和节点j之间的距离
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                self.dist_matrix[i][j] = np.linalg.norm(self.node_pos[i] - self.node_pos[j])

        # 生成动态链路参数（添加随机噪声）
        self.delay_matrix = self.dist_matrix * 0.1 + np.random.normal(0, 0.1, (NUM_NODES, NUM_NODES))
        self.signal_matrix = 100 / (self.dist_matrix + 1) + np.random.normal(0, 0.5, (NUM_NODES, NUM_NODES))
        np.fill_diagonal(self.delay_matrix, 0)  # 对角线置零
        np.fill_diagonal(self.signal_matrix, 0)

    def update_environment(self, new_pos=None, noise_scale=0.1):
        """根据新位置或噪声更新环境"""
        if new_pos is not None:  # 若提供新位置，则更新
            self.node_pos = new_pos
        else:  # 否则添加随机扰动
            self.node_pos += np.random.normal(0, noise_scale, self.node_pos.shape)
        # 重新计算距离、延迟和信号矩阵
        self.get_state()

    def get_state(self):
        """构建神经网络输入特征向量"""
        state = []
        for i in range(NUM_NODES):
            # 节点自身特征：坐标 + 到所有其他节点的信号和延迟
            features = [*self.node_pos[i]]  # 坐标x,y
            for j in range(NUM_NODES):
                if j != i:
                    features.append(self.signal_matrix[i][j])
                    features.append(self.delay_matrix[i][j])
            state.extend(features)
        return torch.FloatTensor(state).unsqueeze(0)


class TopologyOptimizer(nn.Module):
    """拓扑优化神经网络模型"""

    def __init__(self, input_dim,num_nodes=NUM_NODES):
        super().__init__()
        self.num_nodes = num_nodes

        # 第一版本的全连接层
        """"self.net = nn.Sequential(  # 按照顺序堆叠各层
            nn.Linear(input_dim, HIDDEN_DIM),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),  # 隐藏层到隐藏层
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, (NUM_NODES - 1) * (NUM_NODES - 1))  # 输出层维度
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, NUM_NODES - 1, NUM_NODES - 1)"""  # 重塑为(batch_size, 非根节点数, 可选父节点数)

    # 2.0版本使用Transformer模型
        # 定义特征维度：每个节点的特征长度
        self.node_feat_dim = input_dim // num_nodes # 例如50/5=10

        # Transformer编码器层配置
        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.node_feat_dim,  # 输入特征维度（每个节点的特征长度）
            nhead=TRANSFORMER_NHEADS,  # 注意力头数
            dim_feedforward = TRANSFORMER_DIM_FF,  # 前馈网络隐藏层维度
            dropout=DROPOUT,
            batch_first = True  # 关键修正
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=TRANSFORMER_LAYERS)

        # 输出层：将Transformer输出映射到目标维度
        self.output_layer = nn.Linear(
            num_nodes * self.node_feat_dim,  # 输入维度为序列展平后的长度
            (num_nodes - 1) * (num_nodes - 1)  # 与原模型输出一致
        )

    def forward(self, x):
        # 输入 x 的形状应为 [batch_size, input_dim]
        batch_size = x.size(0)
        input_dim = x.size(1)

        # 动态计算每个节点的特征维度
        node_feat_dim = input_dim // self.num_nodes

        # 检查输入维度是否合法
        assert input_dim % self.num_nodes == 0, "input_dim必须能被节点数整除"

        # 重塑张量：[batch_size, num_nodes, node_feat_dim]
        x_reshaped = x.view(batch_size, self.num_nodes, node_feat_dim)

        # 调整维度顺序：[num_nodes, batch_size, node_feat_dim]
        # x_reshaped = x_reshaped.permute(1, 0, 2)

        # 通过 Transformer 编码器
        transformer_output = self.transformer_encoder(x_reshaped)

        # 恢复维度并展平
        output = transformer_output.permute(1, 0, 2).contiguous()
        output_flat = output.view(batch_size, -1)

        # 输出层映射
        logits = self.output_layer(output_flat)
        return logits.view(-1, self.num_nodes - 1, self.num_nodes - 1)

class ReinforcementLearning:
    """强化学习训练系统"""

    def __init__(self):
        self.env = NetworkEnvironment()
        input_dim = NUM_NODES * (2 + 2 * (NUM_NODES - 1))  # 计算输入维度
        # self.model = TopologyOptimizer(input_dim)
        # self.model.parameters()：模型的待优化参数（权重和偏置）。
        # lr=LEARNING_RATE：学习率，控制参数更新步长（例如0.001）。
        self.model = TopologyOptimizer(input_dim)  # 使用新的Transformer模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def select_action(self, logits):
        """根据概率分布选择父节点（修正维度问题）"""
        logits = logits.squeeze(0)  # 移除批次维度 [1,n,n] -> [n,n]
        non_root_nodes = list(range(1, NUM_NODES))
        parent_selections = {}
        log_probs = []

        for idx, node_id in enumerate(non_root_nodes):
            allowed_parents = [n for n in range(NUM_NODES) if n != node_id]

            # 创建有效概率分布（修正维度）
            node_logits = logits[idx][:len(allowed_parents)]  # 确保长度匹配
            # node_logits 是神经网络输出的某个节点的概率分布
            dist = Categorical(logits=node_logits)
            action = dist.sample()  # 采样一个动作（父节点编号）
            parent = allowed_parents[action.item()]  # 正确转换为标量

            parent_selections[node_id] = parent
            # action：选择的父节点编号（整数）。
            # log_prob：该动作的对数概率（用于计算损失函数）。
            log_probs.append(dist.log_prob(action))

        return parent_selections, torch.stack(log_probs)

    def calculate_reward(self, parent_selections):
        """计算奖励值（总延迟信号强度和带宽利用率）"""
        total_delay = 0
        total_signal = 0
        for child, parent in parent_selections.items():
            total_delay += self.env.delay_matrix[child][parent]
            total_signal += self.env.signal_matrix[child][parent]
        # 定义权重（需根据实际需求调整）
        delay_weight = -0.6  # 延迟越小越好
        signal_weight = 0.4  # 信号越大越好
        reward = delay_weight * total_delay + signal_weight * total_signal
        return reward

    def train(self):
        """训练主循环"""
        for episode in range(NUM_EPISODES):
            state = self.env.get_state()
            logits = self.model(state)
            parent_selections, log_probs = self.select_action(logits)
            reward = self.calculate_reward(parent_selections)

            loss = (-torch.sum(log_probs) * reward).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 100 == 0:
                # 每100轮更新一次环境，模拟实时变化，提升动态优化能力
                self.env.update_environment(noise_scale=0.05)
                print(f"Episode {episode:4d} | Total Delay: {-reward:.2f}")

    def test(self):
        """测试训练好的模型"""
        state = self.env.get_state()
        with torch.no_grad():
            logits = self.model(state)
        parent_selections, _ = self.select_action(logits)

        # print("\n优化后的网络拓扑：")
        print("子节点 -> 父节点 | 信号强度 | 链路延迟")
        for child, parent in parent_selections.items():
            print(f"Node {child:2d} -> Node {parent:2d} | "
                  f"Signal: {self.env.signal_matrix[child][parent]:5.2f} | "
                  f"Delay: {self.env.delay_matrix[child][parent]:5.2f}")


if __name__ == "__main__":
    rl = ReinforcementLearning()
    rl.train()
    rl.test()
