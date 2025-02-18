import torch
import numpy as np

device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()  # Clear any cached memory on CUDA

################################## preprocess functions ######################

def feature_norm_2D(x, exCept=None):
    '''
    Normalize a 2D tensor across the last dimension (features).
    :param x: Input tensor of shape [batch, features]
    :param exCept: List of indices to skip during normalization (optional)
    :return: Normalized tensor
    '''
    if x.shape[0] == 1:
        return x
    else:
        f_len = x.size(-1)
        norm_x = torch.empty_like(x)
        for i in range(f_len):
            if exCept == None or exCept[i] == 0:
                feature = x[:, i]
                mean = feature.mean()
                std = feature.std()
                norm_x[:, i] = (feature - mean) / (std + 1e-6)
            else:
                norm_x[:, i] = x[:, i]
        return norm_x

def feature_norm_3D(x, exCept=None):
    '''
    Normalize a 3D tensor across the last dimension (features).
    :param x: Input tensor of shape [batch, seq_len, features]
    :param exCept: List of indices to skip during normalization (optional)
    :return: Normalized tensor
    '''
    if x.shape[1] and x.shape[0] == 1:
        return x
    else:
        f_len = x.size(-1)
        norm_x = torch.empty_like(x)
        for i in range(f_len):
            if exCept == None or exCept[i] == 0:
                feature = x[:,:,i]
                mean = feature.mean()
                std = feature.std()
                norm_x[:,:,i] = (feature - mean) / (std + 1e-5)
            else:
                norm_x[:,:,i] = x[:,:,i]
        return norm_x

def get_job_adjacency(stage_config):
    '''
    Generate the adjacency matrix for jobs based on stage configuration.
    :param stage_config: A list of machine counts at each stage
    :return: Job adjacency matrix (connecting jobs across stages)
    '''
    totalMachineNum = 0
    for i in stage_config:
        totalMachineNum += i  # Sum the total number of machines across all stages

    # Create stage indices
    stage_index = np.zeros_like(stage_config)
    for i in range(len(stage_config)):
        stage_index[i] = sum(stage_config[0:i])

    # Initialize the adjacency matrix
    metrix = np.zeros((totalMachineNum, totalMachineNum))
    for idx, i in enumerate(stage_config[0:len(stage_config)-1]):
        next_stage_num = stage_config[idx + 1]
        next_stage_idx = stage_index[idx + 1]
        for j in range(i):
            for k in range(next_stage_num):
                metrix[stage_index[idx] + j, next_stage_idx + k] = 1
    return metrix

def get_mf_adj(stage_config):
    '''
    Generate machine feature adjacency matrix, considering buffer nodes.
    :param stage_config: A list of machine counts at each stage
    :return: Adjacency matrix including buffer nodes
    '''
    totalMachineNum = 0
    for i in stage_config:
        totalMachineNum += i

    nodes_len = totalMachineNum + len(stage_config)  # Include buffer nodes
    metrix = np.zeros((nodes_len, nodes_len))  # Initialize adjacency matrix
    buffer_index = np.zeros_like(stage_config)

    tmp_stage_config = []
    for i in range(len(stage_config)):
        tmp_stage_config.append(stage_config[i] + 1)
        buffer_index[i] = sum(tmp_stage_config[0:i])

    for idx, i in enumerate(stage_config[0: len(stage_config)]):
        current_buffer_index = buffer_index[idx]
        if(idx < len(stage_config) - 1 ):
            next_buffer_index = buffer_index[idx + 1]
            for k in range(i):
                metrix[int(current_buffer_index), int(current_buffer_index + k + 1)] = 1
                metrix[int(current_buffer_index + k + 1), int(next_buffer_index)] = 1
        else:
            for k in range(i):
                metrix[int(current_buffer_index), int(current_buffer_index + k + 1)] = 1
    return metrix

def set_job_input(job_features, adjacency):
    '''
    :param job_features: a list, indexed by "stage", value: list [jobs, machines (nodes), features];
    :param adjacency: a tensor, [nodes, nodes]
    :return:  {"feature": a list indexed by "stage", value: tensor [batch, jobs, nodes, features],
                "adjacency": a list indexed by stage, value: a tensor [batch, jobs, nodes, nodes],
                "mask": None}
    '''
    features = []
    for i in job_features:
        # Convert each job feature array to a PyTorch tensor and move it to the appropriate device
        tmp_i = torch.tensor(i, dtype=torch.float32).to(device).unsqueeze(0)
        features.append(tmp_i)

    # Note: Create adjacency list indexed by "stages" with dimension [jobs, nodes, nodes]
    adj_list = []
    for i in range(len(job_features)):
        # Expand adjacency to a list based on the number of jobs (jobs dimension) per stage, creating [jobs, nodes, nodes]
        tmp_adj = adjacency
        job_num = features[i].size(1)  # Number of jobs in this stage
        tmp_adj = tmp_adj.unsqueeze(0).repeat(job_num, 1, 1)  # Repeat adjacency matrix for each job
        tmp_adj = tmp_adj.unsqueeze(0)
        adj_list.append(tmp_adj)

    output = {"feature": features, "adjacency": adj_list, "mask": None}
    return output

def set_machine_input(machine_features, adjacency):
    '''
    :param machine_features: a list [nodes, features]
    :param adjacency: a tensor, [nodes, nodes]
    :return: a dict, features: tensor [batch, nodes, features] , adjacency: tensor [batch, nodes, nodes]
    '''
    features = torch.tensor(machine_features, dtype=torch.float32).to(device).unsqueeze(0)
    adjacency = adjacency.unsqueeze(0)  # Add batch dimension
    output = {"feature": features, "adjacency": adjacency, "mask": None}
    return output

def set_system_input(system_features):
    '''
    :param system_features: a list
    :return: [batch, seq_len]
    '''
    if system_features is None:
        return None
    else:
        output = torch.tensor(system_features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        return output

def set_seq_input(action_seq_input):
    # Process the end-to-end action sequence, add it to a dictionary, and add a mask
    action_seq_input = torch.tensor(action_seq_input, dtype=torch.float32).to(device)  # [1, seq, feature]
    feature_len = action_seq_input.shape[-1]
    exCept_index = torch.zeros(feature_len).to(device)
    exCept_index[0] = 1  # Mark the first feature index
    action_dict = {"feature": action_seq_input, "mask": None}
    return action_dict

def set_jm_input(job_feature_input, job_adj):
    # Process job feature input for machine nodes [machine, nodes, feature]
    job_feature_input = torch.tensor(job_feature_input, dtype=torch.float32).to(device)
    machine_num = job_feature_input.shape[0]  # Number of machines
    job_feature_input = job_feature_input.unsqueeze(0)  # Add batch dimension, resulting in [b, machines, nodes, feature]
    tmp_job_adj = job_adj.unsqueeze(0).repeat(machine_num, 1, 1).unsqueeze(0)  # Repeat adjacency for each machine
    job_input_2 = {"feature": job_feature_input, "adjacency": tmp_job_adj}
    return job_input_2

def get_input_state(job_features, job_adj, machine_features, machine_adj, action_seq_input, job_feature_input, system_input):
    job_input = set_job_input(job_features, job_adj)
    machine_input = set_machine_input(machine_features, machine_adj)
    system_input = set_system_input(system_input)
    action_dict = set_seq_input(action_seq_input)
    job_input_2 = set_jm_input(job_feature_input, job_adj)
    state = {"job input": job_input, "machine input": machine_input, "sys input": system_input,
             "action seq input": action_dict, "job input 2": job_input_2}
    return state