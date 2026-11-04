import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class Action_Net(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Action_Net, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.max_action * torch.tanh(self.l4(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 1)

        self.l6 = nn.Linear(state_dim + action_dim, 256)
        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 256)
        self.l9 = nn.Linear(256, 256)
        self.l10 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = self.l5(q1)
        q2 = F.relu(self.l6(sa))
        q2 = F.relu(self.l7(q2))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = self.l10(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = self.l5(q1)
        return q1

    def min_q(self,state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = self.l5(q1)
        q2 = F.relu(self.l6(sa))
        q2 = F.relu(self.l7(q2))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = self.l10(q2)
        q=torch.min(q1,q2)
        return q

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state,action], 1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, z, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state,z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class TD3_BC(object):
    def __init__(self, state_dim, action_dim, max_action, device,
                 discount, tau, policy_noise, noise_clip, policy_freq,actor_lr,value_lr, critic_lr,
                 delta_alpha,alpha,bc_weight,vae_weight,adv_threshold,Is_SubOptimal):
        self.device = device
        self.actor = Action_Net(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_dim,action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.dataset_Is_SubOptimal=Is_SubOptimal


        # NewAPPO独有参数
        self.bc_weight  = bc_weight
        self.vae_weight = vae_weight
        self.alpha = alpha
        self.delta_alpha=delta_alpha
        self.adv_threshold = adv_threshold

        # VAE
        self.valueNet = ValueNet(state_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.valueNet.parameters(), lr=value_lr)
        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        self.vae_weight = vae_weight

        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "value": self.valueNet,
            "value_optimizer": self.value_optimizer,
            "vae": self.vae,
            "vae_optimizer": self.vae_optimizer,
        }

        print("state_dim:", state_dim, "action_dim:", action_dim)

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1
        tb_statics = dict()

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target part so we should use torch.no_grad() to clear the gradient
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        tb_statics.update({"critic_loss": critic_loss.item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # V(s)价值函数
        with torch.no_grad():
           current_Q = self.critic.min_q(state, action)         # 执行动作后的Q值
        value = self.valueNet(state)                            # 当前时刻状态价值
        value_loss = F.mse_loss(value, current_Q.detach())      # 跟踪执行动作后的Q值，得到下一时刻的状态价值
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # VAE生成器
        q_vae= self.critic.Q1(state, action)
        recon, z, mean, std= self.vae(state,action)
        adv_vae  = q_vae-value.detach()
        adv_weight = torch.clamp(adv_vae, min=0.0)
        adv_weight = adv_weight / (adv_weight.mean() + 1e-6)
        adv_weight = adv_weight.clamp(0.0, 20)
        adv_weight = adv_weight * self.vae_weight
        recon_loss = (adv_weight * torch.square(recon - action.detach())).mean()
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        tb_statics.update({"vae_loss": vae_loss.item(),
                           "recon_loss": recon_loss.item(),
                           "KL_loss": KL_loss.item(),
                           })

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q_pi = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q_pi.abs().mean().detach()
            actor_loss = -lmbda * Q_pi.mean()

            #1. action中的优势
            Q_new = self.critic.Q1(state, action)
            advantage_action = Q_new - value

            #2. 重构recon_pi中的优势
            recon_pi, _, _,_= self.vae(state, pi)
            Q_recon_pi = self.critic.Q1(state, recon_pi)
            advantage_recon_pi = Q_recon_pi - value

            #3. 重构recon_ac中的优势
            recon_ac, _, _,_= self.vae(state, action)
            Q_ac= self.critic.Q1(state, recon_ac)
            advantage_ac = Q_ac - value

            if self.total_it>50000:
                #4.分别向action、pi中加入delta
                delta1 = self.Q_adversarial_delta(state, action.detach(), delta_alpha=self.delta_alpha)
                action_delta = action + delta1
                recon_action_delta, _, _, _    = self.vae(state, action_delta)
                recon_action_delta=self.suppportCons(action_delta, recon_action_delta, action)
                Q_recon_action_delta           = self.critic.Q1(state, recon_action_delta)
                Advantage_recon_action_delta   = Q_recon_action_delta-value

                delta2 = self.Q_adversarial_delta(state, pi.detach(), delta_alpha=self.delta_alpha)
                pi_delta = pi + delta2
                recon_pi_delta, _, _, _ = self.vae(state, pi_delta)
                recon_pi_delta = self.suppportCons(pi_delta, recon_pi_delta, pi)
                Q_recon_pi_delta = self.critic.Q1(state, recon_pi_delta)
                Advantage_recon_pi_delta = Q_recon_pi_delta-value

                #集成为一个矩阵，求dim=0上的最大值
                stacked_adv = torch.stack([advantage_action,advantage_recon_pi, advantage_ac,\
                                           Advantage_recon_pi_delta,Advantage_recon_action_delta], dim=0)
                max_values  = torch.max(stacked_adv, dim=0)[0]
                max_indices = torch.max(stacked_adv, dim=0)[1]

                # 找到embedding-->behavior policy
                embedding_action = action * (advantage_action == max_values)
                embedding_recon_pi = recon_pi * (advantage_recon_pi == max_values)
                embedding_recon_action =  recon_ac * (advantage_ac == max_values)
                embedding_recon_pi_delta = recon_pi_delta * (Advantage_recon_pi_delta == max_values)
                embedding_recon_action_delta = recon_action_delta * (Advantage_recon_action_delta == max_values)
                embedding =  embedding_action +embedding_recon_pi + embedding_recon_action+ \
                             embedding_recon_pi_delta+embedding_recon_action_delta
            else:
                # 集成为一个矩阵，求dim=0上的最大值
                stacked_adv = torch.stack([advantage_action, advantage_recon_pi, advantage_ac], dim=0)
                max_values = torch.max(stacked_adv, dim=0)[0]
                max_indices = torch.max(stacked_adv, dim=0)[1]

                # 找到embedding-->behavior policy
                embedding_action = action * (advantage_action == max_values)
                embedding_recon_pi = recon_pi * (advantage_recon_pi == max_values)
                embedding_recon_action = recon_ac * (advantage_ac == max_values)
                embedding = embedding_action + embedding_recon_pi + embedding_recon_action

            # 计算优势Mask
            mask_pos = (max_values > self.adv_threshold)
            Q_weight = mask_pos.int().detach() * self.bc_weight

            # Optimize the actor
            bc_loss = (torch.square(pi - embedding.detach()) * Q_weight).mean()
            combined_loss = actor_loss + bc_loss
            self.actor_optimizer.zero_grad()
            combined_loss.backward()
            self.actor_optimizer.step()

            tb_statics.update(
                {   "Adversarial_sum":[(max_indices==i).sum().item() for i in range(0,stacked_adv.shape[0])],
                    "Adv_threshold":[(max_indices[mask_pos]==i).sum().item() for i in range(0,stacked_adv.shape[0])],
                    # "adv_delta": [delta3[mask_pos.squeeze(-1) & (Advantage_recon_pi_delta == max_values).squeeze(-1)]],
                    "mask_num":  mask_pos.int().sum().item(),
                    "bc_loss": bc_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "combined_loss": combined_loss.item(),
                    "Q_value": torch.mean(Q_pi).item(),
                    "target_Q": torch.mean(target_Q).item(),
                    "lmbda": lmbda,
                }
            )

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1-self.tau) * target_param.data
                )
        return tb_statics

    def save(self, model_path):
        state_dict = dict()
        for model_name, model in self.models.items():
            state_dict[model_name] = model.state_dict()
        torch.save(state_dict, model_path)

    def load(self,model_path):
        state_dict = torch.load(model_path)
        for model_name, model in self.models.items():
            model.load_state_dict(state_dict[model_name])

    def suppportCons(self,perturbed_action,vae_recon,action,support_threshold=0.1): # VAE支持距离
        support_distance = torch.sum(
            (perturbed_action - vae_recon) ** 2,
            dim=-1,
            keepdim=True
        )
        support_mask = (
                support_distance < support_threshold
        ).float()

        # 只保留on-support扰动动作
        final_action = (
                support_mask * perturbed_action
                + (1 - support_mask) * action
        )
        return final_action

    def Q_adversarial_delta(self, state, expert_action,delta_alpha,epsilon=0.1,num_iter=1):
        expert_action.requires_grad = True
        delta = torch.zeros_like(expert_action, requires_grad=True)
        perturbed_pi = expert_action + delta
        for _ in range(num_iter):
            if self.dataset_Is_SubOptimal is True:
                Q_value=self.critic.min_q(state,perturbed_pi)
            else:
                Q_value = self.critic.Q1(state, perturbed_pi)
            self.critic.zero_grad()
            # 定义损失loss1
            loss=Q_value.mean()
            grad=torch.autograd.grad(loss,perturbed_pi)[0] # 对perturbed_pi求导
            grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-8
            perturbed_pi = perturbed_pi + delta_alpha * (grad / grad_norm)  # 使用+号 梯度上升,loss增大，使得Q值/adv优势值增大
            #裁剪操作
            delta = torch.clamp(perturbed_pi-expert_action, -epsilon, epsilon)
            perturbed_pi = torch.clamp(expert_action + delta, -self.max_action, self.max_action)
        delta=perturbed_pi-expert_action
        return delta.detach()