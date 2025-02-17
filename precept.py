from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from net import GCNLayer, GATLayer, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Job_encoder(nn.Module):
    def __init__(self, par):
        super(Job_encoder, self).__init__()
        self.par = par
        self.layer_num = self.par['layer_num']  # Number of layers in the GAT network

        # Initializing the GAT layers with respective input and output feature dimensions
        self.GAT_1 = GATLayer(self.par['job feature'], self.par['job hidden'])
        self.GAT_2 = GATLayer(self.par["job hidden"], self.par["job hidden"])
        self.GAT_3 = GATLayer(self.par["job hidden"], self.par["job hidden"])
        self.GAT_4 = GATLayer(self.par["job hidden"], self.par["job hidden"])

    def forward(self, adjacency, feature, mask):
        # Encoding the jobs of one stage into a hidden vector using GAT layers
        job_tmp_1 = self.GAT_1(adjacency, feature)
        job_tmp_1 = torch.tanh(job_tmp_1)
        job_tmp_2 = self.GAT_2(adjacency, job_tmp_1)
        job_tmp_2 = job_tmp_2 + job_tmp_1
        job_tmp_2 = torch.tanh(job_tmp_2)
        job_tmp_3 = self.GAT_3(adjacency, job_tmp_2)
        job_tmp_3 = job_tmp_3 + job_tmp_2
        job_tmp_3 = torch.tanh(job_tmp_3)
        job_tmp_4 = self.GAT_4(adjacency, job_tmp_3)
        job_tmp_4 = job_tmp_3 + job_tmp_4
        job_tmp_4 = torch.tanh(job_tmp_4)

        # Pooling step:
        output = torch.mean(job_tmp_4, dim=-2)
        return output

class Precept(nn.Module):
    def __init__(self, par):
        super(Precept, self).__init__()
        self.par = par
        self.layer_num = self.par['layer_num']

        # step1: job feature encoding
        self.j_encoder = Job_encoder(self.par)
        self.lstm = nn.LSTM(self.par['job hidden'], self.par['lstm hidden'], batch_first=True)
        self.norm_1 = LayerNorm(self.par['lstm hidden'])

        # step2: concat job feature and machine feature, embedding them
        self.machine_first_emb = nn.Linear(self.par['machine feature'], self.par['lstm hidden'])
        self.norm_2 = LayerNorm(self.par['lstm hidden'])

        # Determine input dimension for machine embedding
        if not self.par['use posterior probability']:
            self.machine_emb_input_dim = self.par['lstm hidden'] * 2
        else:
            self.machine_emb_input_dim = self.par['lstm hidden'] * 2 + self.par['posterior probability dim']

        self.machine_feature_emb = nn.Linear(self.machine_emb_input_dim, self.par['machine emb dim'])
        self.norm_3 = LayerNorm(self.par['machine emb dim'])

        # Create GAT layers and their corresponding normalization layers in a loop
        self.GAT_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_dim = self.par['machine emb dim']
        for i in range(4):  # Assuming 4 GAT layers, adjust if needed
            self.GAT_layers.append(GATLayer(in_dim, self.par['machine GNN hidden']))
            self.norm_layers.append(LayerNorm(self.par['machine GNN hidden']))
            in_dim = self.par['machine GNN hidden']

    def forward(self, machine_input, job_input, sys_input, job_input_2):

        j_stages_f = job_input['feature']
        j_stages_a = job_input['adjacency']
        j_mask = job_input['mask']
        stage_num = len(j_stages_f)
        j_encoded = []

        # Step 1: Encode job features for each stage using the job encoder and LSTM
        for i in range(stage_num):
            jobs_feature = j_stages_f[i]
            jobs_adjacency = j_stages_a[i]
            if "packing" in job_input:
                jobs_packing = True
                jobs_length = job_input["length"][i]
                jobs_mask = j_mask[i]
            else:
                jobs_packing = False
                jobs_length = None
                jobs_mask = j_mask
            job_emb = self.j_encoder(jobs_adjacency, jobs_feature, jobs_mask)

            # LSTM encoding job_emb to a vector
            if jobs_packing:
                packed = pack_padded_sequence(job_emb, jobs_length, batch_first=True, enforce_sorted=False)
                _, (job_context, _) = self.lstm(packed)
            else:
                _, (job_context, _) = self.lstm(job_emb)
            job_context = job_context[-1]  # Take the final hidden state
            j_encoded.append(job_context)

        # Step 1.2: Encode job features on machine nodes for later concatenation
        j_machine_f = job_input_2["feature"]
        j_machine_a = job_input_2["adjacency"]
        j_machine_encoded = self.j_encoder(j_machine_a, j_machine_f, None)
        j_machine_encoded = j_machine_encoded.permute(1, 0, 2)

        # Step 2.1: Process machine features and concatenate with job features
        machine_f = machine_input["feature"]
        machine_a = machine_input["adjacency"]
        machine_mask = machine_input["mask"]
        machine_feature_list = []

        # Create an index to distinguish stage nodes (1) from machine nodes (0)
        stage_index = (machine_f.permute(1, 0, 2) > 0).float()
        machine_f = self.machine_first_emb(machine_f)
        machine_f = self.norm_2(machine_f, machine_mask)
        machine_f = torch.tanh(machine_f)
        machine_f = machine_f.permute(1, 0, 2)

        machine_index = 0
        node_index = 0
        for i in machine_f:
            if stage_index[node_index][0][0] > 0:
                # Concatenate machine features with encoded job features for stage nodes
                machine_feature = torch.cat((i, j_encoded.pop(0)), dim=1)
                machine_feature_list.append(machine_feature)
            else:
                # Concatenate machine features with encoded job features for machine nodes
                j_m_feature = j_machine_encoded[machine_index]
                machine_index = machine_index + 1
                machine_feature = torch.cat((i, j_m_feature), dim=1)
                machine_feature_list.append(machine_feature)
            node_index += 1

        # Step 2.2: Stack and embed the concatenated features
        nodes_feature = torch.stack(machine_feature_list)
        nodes_feature = nodes_feature.permute(1, 0, 2)
        nodes_feature = self.machine_feature_emb(nodes_feature)
        nodes_feature = self.norm_3(nodes_feature, None)
        nodes_feature = torch.tanh(nodes_feature)

        # Step 3: Apply GAT layers with residual connections and normalization
        output_feature = nodes_feature  # Initial input to the GAT layers

        for i in range(4):  # Loop over 4 GAT layers
            # Apply GAT layer and LayerNorm
            output_feature = self.GAT_layers[i](machine_a, output_feature)
            output_feature = self.norm_layers[i](output_feature)  # Normalize the output

            # Add residual connection from the previous layer (except the first layer)
            if i != 0:
                output_feature = output_feature + prev_feature

            # Apply tanh activation
            output_feature = torch.tanh(output_feature)

            # Save the current output for the residual connection in the next iteration
            prev_feature = output_feature

        # Step 4: Aggregate node features by taking the mean across nodes
        output = torch.mean(output_feature, dim=1)
        return output