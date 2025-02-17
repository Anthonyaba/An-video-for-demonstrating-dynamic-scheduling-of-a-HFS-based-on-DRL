import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import os
import pickle as pkl
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from net import GCNLayer, GATLayer, LayerNorm
import random
from preprocess_functions import *
from precept import *
################################## set net parameters #############################
def ini_net_par(job_feature, machine_feature, use_graph, state_dim, action_dim, hidden_size, seq_dim):
    '''
    Initializes the network parameters.
    :param job_feature: job feature length
    :param machine_feature: machine feature length
    :return: Dictionary containing network parameters
    '''
    net_par = {
        "job feature": job_feature,  # Job feature length
        "job hidden": hidden_size,   # Hidden layer size for job features
        "lstm hidden": hidden_size,  # Hidden layer size for LSTM
        "use posterior probability": False,  # Whether to use posterior probability
        "machine feature": machine_feature,  # Machine feature length
        "machine emb dim": hidden_size,  # Machine embedding dimension
        "machine GNN hidden": hidden_size,  # Hidden size for machine GNN
        "use graph": use_graph,  # Whether to use graph
        "state dim": state_dim,  # State dimension
        "action dim": action_dim,  # Action dimension
        "decoder hidden size": hidden_size,  # Hidden size for decoder
        "seq feature dim": seq_dim  # Sequence feature dimension
    }
    return net_par

################################## set seed ################################

m_seed = 1
torch.manual_seed(m_seed)  # Set manual seed for reproducibility
torch.cuda.manual_seed(m_seed)  # Set seed for CUDA
np.random.seed(m_seed)  # Set seed for numpy
torch.backends.cudnn.deterministic = True  # Ensure deterministic results for CUDA

################################## set device ################################

print("============================================================================================")
# Set the device to either CPU or CUDA
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()  # Clear any cached memory on CUDA
    print("Device set to : " + str(torch.cuda.get_device_name(device)))  # Print device name
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## PPO Policy ################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.discount_reward = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.discount_reward[:]

    def save(self):
        dict = {'actions': self.actions, 'states': self.states, 'logprobs': self.logprobs,
                'rewards': self.logprobs, 'is_terminals': self.is_terminals}
        with open(self.buffer_path, 'wb') as f:
            pkl.dump(dict, f)

    def delete(self):
        self.actions.pop()
        self.states.pop()
        self.logprobs.pop()
        self.rewards.pop()
        self.is_terminals.pop()
        self.discount_reward.pop()
    def sample(self, batch_size, rewards):
        indices = random.sample(range(len(self.states)), batch_size)
        sample_state = [self.states[i] for i in indices]
        sample_action = [self.actions[i] for i in indices]
        sample_logprobs = [self.logprobs[i] for i in indices]
        sample_rewards = [rewards[i] for i in indices]
        return sample_state, sample_action, sample_logprobs, sample_rewards

class Actor(nn.Module):
    def __init__(self, par):
        super(Actor, self).__init__()
        self.par = par
        self.decoder_hidden_size = self.par["decoder hidden size"]
        self.action_dim = self.par["action dim"]
        self.precept = Precept(par=self.par)
        self.context_emb = nn.Linear(self.par['machine GNN hidden'], self.decoder_hidden_size)
        self.seq_emb = nn.Linear(self.par["seq feature dim"], self.decoder_hidden_size)
        self.action_seq_emb = nn.Linear(self.decoder_hidden_size, 1)

    def forward(self, state):
        job_input = state["job input"]
        machine_input = state["machine input"]
        sys_input = state["sys input"]
        job_input_2 = state["job input 2"]
        input_seq = state["action seq input"]["feature"]
        mask = state["action seq input"]["mask"]

        input_seq = self.seq_emb(input_seq)
        context = self.precept(machine_input, job_input, sys_input, job_input_2)

        # Global feature embedding: Map context to decoder hidden size
        context = self.context_emb(context)

        # Concatenate input sequence and global context using element-wise multiplication
        action_seq = torch.einsum("bh, bsh -> bs", context, input_seq)

        # Handle valid actions using the mask, and apply softmax to get action probabilities
        if mask is not None:
            valid_mask = ~mask
            action_seq = action_seq * valid_mask + (-1e9) * mask
            action_probs = F.softmax(action_seq, dim=-1)
        else:
            action_probs = F.softmax(action_seq, dim=-1)

        return action_probs  # Return the action probability distribution

class Critic(nn.Module):
    def __init__(self, par):
        super(Critic, self).__init__()

        self.par = par
        self.decoder_hidden_size = self.par["decoder hidden size"]
        self.action_dim = self.par["action dim"]
        self.precept = Precept(par=self.par)
        self.context_emb = nn.Linear(self.par['machine GNN hidden'], self.decoder_hidden_size)
        self.seq_emb = nn.Linear(self.par["seq feature dim"], self.decoder_hidden_size)
        self.action_seq_emb = nn.Linear(self.decoder_hidden_size, 1)
        self.norm = LayerNorm(self.decoder_hidden_size)

    def forward(self, state):
        # Extract inputs from the state dictionary
        machine_input = state["machine input"]
        job_input = state["job input"]
        sys_input = state["sys input"]
        job_input_2 = state["job input 2"]
        input_seq = state["action seq input"]["feature"]
        mask = state["action seq input"]["mask"]

        # Embed the input sequence
        input_seq = self.seq_emb(input_seq)

        # Get the context features from Precept
        context = self.precept(machine_input, job_input, sys_input, job_input_2)

        # Embed the context features
        context = self.context_emb(context)

        # Combine the context and input sequence using a weighted sum
        action_seq = torch.einsum("bh, bsh -> bs", context, input_seq)

        # Compute the average value for the valid part of the sequence
        if mask is not None:
            mask = ~mask
            action_seq = action_seq * mask
            sum = action_seq.sum(dim=1, keepdim=True)
            count = mask.sum(dim=1, keepdim=True)
            value = sum / count
        else:
            value = torch.mean(action_seq, dim=-1)

        return value


class ActorCritic(nn.Module):
    def __init__(self, par, train):
        super(ActorCritic, self).__init__()
        self.Train = train
        self.par = par
        # actor
        self.actor = Actor(self.par)
        # critic
        self.critic = Critic(self.par)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        self.actor.eval()
        action_probs = self.actor(state)
        if self.Train:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            dist = Categorical(action_probs)
            action = torch.argmax(action_probs)
            action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_size,
                 job_f, machine_f, seq_f, stage_config, use_graph, episode_index, train, exp_index, exist_model=False):
        self.episode_index = episode_index
        self.par = ini_net_par(job_feature=job_f, machine_feature=machine_f, use_graph=use_graph
                               , state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size,
                               seq_dim=seq_f)
        self.global_reward = 1
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        if use_graph == True:
            self.job_adjacency = torch.from_numpy(get_job_adjacency(stage_config)).float().to(device)
            self.machine_adjacency = torch.from_numpy(get_mf_adj(stage_config)).float().to(device)

        self.model_path, self.loss_save_path, self.best_path = self.creat_checkpoint_path(exp_index)
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.par, train).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = ActorCritic(self.par, train).to(device)
        self.policy_old.eval()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.loss = []
        if exist_model == True:
            self.load(self.model_path)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        print("Trajectory length： ", len(self.buffer.actions))
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, d_reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.discount_reward), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + d_reward + (self.gamma * discounted_reward)
            final_reward = discounted_reward
            rewards.insert(0, final_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_job_input, old_machine_input, old_system_input, action_input, old_job_input_2 = self.batch_process(
            self.buffer.states)
        old_states ={"job input": old_job_input, "machine input": old_machine_input, "sys input": old_system_input,
                     "action seq input": action_input, "job input 2": old_job_input_2}
        old_actions =  torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self):
        # save model parameters
        torch.save(self.policy_old.state_dict(), self.model_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        
    def creat_checkpoint_path(self,exp_index):

        directory = "GNN/"

        checkpoint_path = directory + "PPO{}.pth".format(exp_index)
        loss_save_path = directory + "PPO_loss{}.pkl".format(exp_index)
        best_path = directory + "PPO_best{}.pth".format(exp_index)
        print("save operation select policy checkpoint path : " + checkpoint_path)
        return checkpoint_path, loss_save_path, best_path

    def save_best(self,path,best):
        data = pd.read_csv(path)
        data = np.array(data)
        mean1 = data[-50:].mean()
        if(mean1<best):
            torch.save(self.policy_old.state_dict(), self.best_path)
            return mean1
        else:
            return best

    def batch_process(self, sample_states):
        batch_len = len(sample_states)
        stage_num = len(sample_states[0]["job input"]["feature"])
        batch_padded_features = []
        batch_padded_adjacency = []
        stage_seq_len = []
        mask_list = []

        # Step 1: Batch process job input features and adjacency matrices
        for s in range(stage_num):
            stage_batch_feature = []
            stage_batch_adjacency = []
            seq_len = []
            for b in range(batch_len):
                tmp_featrue = sample_states[b]["job input"]["feature"][s]
                tmp_featrue = tmp_featrue.squeeze(0)
                stage_batch_feature.append(tmp_featrue)
                tmp_adj = sample_states[b]["job input"]["adjacency"][s]
                tmp_adj = tmp_adj.squeeze(0)
                stage_batch_adjacency.append(tmp_adj)
                seq_len.append(tmp_featrue.shape[0])
            stage_padded_f = pad_sequence(stage_batch_feature, batch_first=True)
            batch_padded_features.append(stage_padded_f)
            stage_padded_a = pad_sequence(stage_batch_adjacency, batch_first=True)
            batch_padded_adjacency.append(stage_padded_a)
            stage_seq_len.append(seq_len)
            # Create mask for padded sequences
            max_job_num = stage_padded_f.shape[1]
            mask = torch.zeros((batch_len, max_job_num)).to(device)
            for index, l in enumerate(seq_len):
                mask[index, :l] = 1  # 1 indicates valid data, 0 indicates padding
            mask_list.append(mask)
        old_job_input = {"feature": batch_padded_features, "adjacency": batch_padded_adjacency,
                         "packing": True, "length": stage_seq_len, "mask": mask_list}

        # Step 2: Batch process machine input, system input, and job features for machine nodes
        batch_m_f = []
        batch_m_a = []
        j_m_f = []
        j_m_a = []
        for b in range(batch_len):
            tmp_machine_feature = sample_states[b]["machine input"]["feature"]
            tmp_machine_adjacency = sample_states[b]["machine input"]["adjacency"]
            tmp_machine_feature = tmp_machine_feature.squeeze(0)
            tmp_machine_adjacency = tmp_machine_adjacency.squeeze(0)
            batch_m_f.append(tmp_machine_feature)
            batch_m_a.append(tmp_machine_adjacency)

            tmp_j_m_feature = sample_states[b]["job input 2"]["feature"].squeeze(0)
            tmp_j_m_adjacency = sample_states[b]["job input 2"]["adjacency"].squeeze(0)
            j_m_f.append(tmp_j_m_feature)
            j_m_a.append(tmp_j_m_adjacency)
        batch_m_f = torch.stack(batch_m_f)
        batch_m_a = torch.stack(batch_m_a)
        batch_j_m_f = torch.stack(j_m_f)
        batch_j_m_a = torch.stack(j_m_a)
        old_machine_input = {"feature": batch_m_f, "adjacency": batch_m_a, "mask": None}
        old_job_input_2 = {"feature": batch_j_m_f, "adjacency": batch_j_m_a}
        batch_sys_f = []
        if sample_states[0]["sys input"] is not None:
            for b in range(batch_len):
                tmp_sys_feature = sample_states[b]["sys input"]
                tmp_sys_feature = tmp_sys_feature.squeeze(0)
                batch_sys_f.append(tmp_sys_feature)
            batch_sys_f = torch.stack(batch_sys_f)
            old_system_input = batch_sys_f
        else:
            old_system_input = None

        # Step 3: Batch process action sequence input (operations)
        old_action_seq = []
        action_seq_len = []
        for b in range(batch_len):
            tmp_feature = sample_states[b]["action seq input"]["feature"]
            tmp_feature = tmp_feature.squeeze(0)
            old_action_seq.append(tmp_feature)
            action_seq_len.append(tmp_feature.size()[0])
        padded_action_seq = pad_sequence(old_action_seq, batch_first=True)
        max_seq_len = padded_action_seq.shape[1]
        action_mask = torch.ones((batch_len, max_seq_len)).to(device)  # 1 indicates masked, 0 indicates valid
        for index, l in enumerate(action_seq_len):
            action_mask[index, :l] = 0
        action_mask = action_mask.type(torch.bool)
        action_input = {"feature": padded_action_seq, "mask": action_mask}

        return old_job_input, old_machine_input, old_system_input, action_input, old_job_input_2