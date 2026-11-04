import copy
from   torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.distributions import Normal

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.v = MLP(state_dim, 1)

    def forward(self, state):
        return self.v(state)

# ============================================================
# Deterministic Policy
# ============================================================

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * self.max_action

    def sample(self, state):
        action = self.forward(state)
        return action, None


# ============================================================
# Gaussian Policy
# ============================================================

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,LOG_STD_MIN=-5.0, LOG_STD_MAX=2.0):
        super().__init__()
        self.max_action = max_action
        self.Log_STD_MIN = LOG_STD_MIN
        self.Log_STD_MAX = LOG_STD_MAX
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.Log_STD_MIN, self.Log_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # 高斯空间采样：(-inf, +inf)
        action = torch.tanh(normal_sample)  # 挤压到：(-1, 1)
        # 计算高斯分布下的 log_prob
        log_prob = dist.log_prob(normal_sample)
        # 同时将求和 (sum) 放到最后一步，让代码更清晰
        jacobian = torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob - jacobian
        # 调整形状为 [batch_size, 1]
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 放大到实际动作空间
        action = action * self.max_action
        return action, log_prob

    def get_log_prob(self, state, action):
        # 1. 确保 action 严格在 (-1, 1) 之间，避免 arctanh 爆发 NaN
        action = torch.clamp(action / self.max_action, -0.999999, 0.999999)

        # 2. 反向换算：把有界动作还原成高斯分布对应的原始值 (raw_action)
        # arctanh(x) = 0.5 * log((1+x)/(1-x))
        raw_action = 0.5 * torch.log((1 + action) / (1 - action))

        # 3. 计算高斯分布下的 log_prob
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)

        # 4. 雅可比行列式修正（注意：减去的是 1 - action^2）
        # 因为 action = tanh(raw_action)，所以 1 - tanh^2(raw_action) == 1 - action^2
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        return log_prob

def expectile_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)

class IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        max_timesteps,
        discount,
        tau,
        expectile,
        temperature,
        critic_lr,
        value_lr,
        actor_lr,
        policy_type="Deterministic"):

        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.DEVICE=device
        self.policy_type = policy_type
        self.total_it = 0

        # Networks
        self.q_net = TwinQ(state_dim, action_dim).to(self.DEVICE)
        self.q_target = copy.deepcopy(self.q_net)
        self.v_net = ValueNetwork(state_dim).to(self.DEVICE)

        if self.policy_type=="Gaussian":
            self.actor = GaussianPolicy(
                state_dim,
                action_dim,
                max_action
            ).to(self.DEVICE)
        elif self.policy_type=="Deterministic":
            self.actor = DeterministicPolicy(
                state_dim,
                action_dim,
                max_action
            ).to(self.DEVICE)
        else:
            raise ValueError(policy_type)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=critic_lr)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=value_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_timesteps)


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    # ========================================================
    # Train
    # ========================================================

    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1
        tb_statics = dict()

        state, action, next_state , reward, not_done = replay_buffer.sample(batch_size)
        # ====================================================
        # Update V via Expectile Regression
        # ====================================================

        with torch.no_grad():
            target_q = self.q_target.min_q(state, action)

        v = self.v_net(state)
        v_loss = expectile_loss(
            target_q - v,
            self.expectile
        ).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # ====================================================
        # Update Q
        # ====================================================

        with torch.no_grad():
            next_v = self.v_net(next_state)
            q_target = reward + not_done * self.discount * next_v

        q1, q2 = self.q_net(state, action)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ====================================================
        # Update Actor
        # ====================================================

        with torch.no_grad():
            q = self.q_net.min_q(state, action)
            v = self.v_net(state)
            adv = q - v
            exp_adv = torch.exp(self.temperature * adv)
            exp_adv = torch.clamp(exp_adv, max=100.0)

        if self.policy_type == "Gaussian":
            log_prob = self.actor.get_log_prob(state, action)
            actor_loss = -(exp_adv * log_prob).mean()

        elif self.policy_type == "Deterministic":
            pred_action = self.actor(state)
            bc_loss = ((pred_action - action) ** 2).mean(dim=-1, keepdim=True)
            actor_loss = (exp_adv * bc_loss).mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        # ====================================================
        # Soft Update
        # ====================================================

        for param, target_param in zip(self.q_net.parameters(),self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data +(1 - self.tau) * target_param.data)

        tb_statics.update({
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item()
        })

        return tb_statics



