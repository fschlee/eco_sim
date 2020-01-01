from eco_sim import Environment # , start_with_gui

import torch
import torch.nn.functional as fun
import numpy
import random
import threading
import time

device=torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device=torch.device("cuda")

def transposed_tensor(arr):
    # (map_height, map_width, max_reps_per_square : D, rep_size : C) => (N, C, H, W, D)
    return torch.from_numpy(numpy.transpose(arr)).to(device)


class ActionValues(torch.nn.Module):
    def __init__(
            self,
            action_space_size=Environment.action_space_size(),
            map_height=Environment.map_height(),
            map_width=Environment.map_width(),
            max_reps_per_square=Environment.max_reps_per_square(),
            rep_size=Environment.rep_size(),
            phys_size=Environment.physical_rep_size(),
            mental_size=Environment.mental_rep_size()):
        super(ActionValues, self).__init__()
        self.h = map_height
        self.w = map_width
        self.d = max_reps_per_square
        self.c0 = rep_size
        # stack (map_height: D, map_width : W, max_reps_per_square : H, rep_size : C)T => (N, C, H, W, D)
        # kernel (h, w, d)
        kernel = (max_reps_per_square, 1, 1)
        self.c1 = 16
        self.c2 = 16
        self.c3 = 8
        hsz = self.c3 * map_height * map_width
        self.state = None
        self.conv1 = torch.nn.Conv3d(rep_size, self.c1, groups=1, kernel_size=kernel)
        #(N, c1, 1, W, D) => (N, c1, W, D)
        self.bn1 = torch.nn.BatchNorm3d(self.c1)
        self.conv2 = torch.nn.Conv2d(self.c1, self.c2, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.c2)
        self.conv3 = torch.nn.Conv2d(self.c2, self.c3, kernel_size=(5, 5), stride=1, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(self.c3)
        ag_size = phys_size + mental_size
        self.ag = torch.nn.Linear(ag_size, ag_size)
        self.lstm = torch.nn.LSTM(self.c3 * self.h * self.w + ag_size, hsz)
        self.act = torch.nn.Linear(hsz, action_space_size)

    def forward(self, x, p, m):
        k = x.shape[0]
        x = fun.relu(self.bn1(self.conv1(x)))
        x = torch.squeeze(x, dim=2)
        x = fun.relu(self.bn2(self.conv2(x)))
        x = fun.relu(self.bn3(self.conv3(x)))
        ag = fun.relu(self.ag(torch.cat((p, m), 1)))
        x, s = self.lstm(torch.cat((x.view(1, k, -1), ag.view(1, k, -1)), 2), self.state)
        self.state = s
        act = self.act(fun.relu(x.view(k, -1)))
        return act

    def reset_state(self):
        self.state = None

    def detach_state(self):
        self.state = (self.state[0].detach(), self.state[1].detach())

    def remap(self, remapping, k):
        after = 0
        for (i, new_i) in remapping:
            if new_i is not None:
                after = new_i + 1
                self.state[0][:, new_i, :] = self.state[0][:, i, :]
                self.state[1][:, new_i, :] = self.state[1][:, i, :]
        if len(remapping) > 0 and k > after:
            s0 = self.state[0].shape
            s1 = self.state[1].shape

            self.state[0][:, after:k, :] = torch.zeros((s0[0], k - after, s0[2]))
            self.state[1][:, after:k, :] = torch.zeros((s0[0], k - after, s1[2]))

class QLearner:
    def __init__(self, env=None, seed=0, pretrained=None, start_gui=False):

        self.env = Environment(seed)
        if start_gui:
            self.env.start_gui()
        self.next_seed = seed + 1
        if pretrained is not None and isinstance(pretrained, ActionValues):
            self.policy = pretrained
        else:
            self.policy = ActionValues()
        self.value = ActionValues().eval()
        self.optimizer = torch.optim.RMSprop(self.policy.parameters())
        self.optimizer

    def run(self, epochs=400, wait=0):
        print("foo")
        time.sleep(wait)
        losses = []
        running_loss = 0.0
        total_count = 0
        action_space = Environment.action_space_size()
        for epoch in range(0, epochs):
            self.policy.reset_state()
            self.value.reset_state()
            current_count = 0
            in_epoch_count = 0
            self.env.reset(self.next_seed)
            self.next_seed += 1
            registered = self.env.register_agents(8)
            k = len(registered)
            obsv, rewards, suggested_actions, physical, mental, visibility, remappings, complete  = self.env.state()
            rewards = torch.tensor(rewards).view(-1, 1)
            done = complete
            rand = random.Random()
            inp = torch.stack([transposed_tensor(np) for np in obsv])
            men = torch.from_numpy(mental).to(device)
            phys = torch.from_numpy(physical).to(device)
            death_count = 0
            while not done and in_epoch_count < 1000:
                if death_count > 0:

                    reg = self.env.register_agents(death_count)
                    k = len(reg)
                    registered = reg
                    self.policy.remap(remappings, k)
                    self.value.remap(remappings, k)
                    tup = self.env.state()
                    rewards = torch.tensor(tup[1]).view(-1, 1)
                total_count += 1
                in_epoch_count += 1
                current_count += 1
                pol = self.policy.forward(inp, phys, men)
                actions = [
                    rand.choices(
                        (rand.choice(range(0, action_space)),
                         suggested_actions[i],
                         torch.argmax(pol[i])),
                        weights=(0.1, 0.3, 0.6),
                        k=1)[0]
                    for i in range(0, k)]
                obsv, new_rewards, new_suggested_actions, physical, mental, visibility, remappings, complete = self.env.step(actions)
                done = complete
                inp = torch.stack([transposed_tensor(np) for np in obsv])
                men = torch.from_numpy(mental).to(device)
                phys = torch.from_numpy(physical).to(device)
                new_rewards = torch.tensor(new_rewards).view(-1, 1)
                expected_rewards = new_rewards - rewards
                death_count = 0
                with torch.no_grad():
                    m,_ = self.value.forward(inp, phys, men).max(1, keepdim=True)
                    for (i, target) in remappings:
                        if target is None:
                            death_count += 1
                            m[i, 0] = 0.0
                    expected_rewards += m
                rewards = new_rewards
                loss = fun.smooth_l1_loss(pol.gather(1, torch.tensor(actions).view(k, 1)), expected_rewards)
                running_loss += loss.item()
                if done or current_count > 20:
                    self.value.load_state_dict(self.policy.state_dict())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    avg_loss = running_loss / current_count
                    print(avg_loss)
                    print(rewards.view(-1))
                    losses.append(avg_loss)
                    current_count = 0
                    running_loss = 0.0
                    self.policy.detach_state()
                else:
                    loss.backward(retain_graph=True)
            torch.save(self.policy, "qlearner.bak")




if __name__ == '__main__':
    QLearner(0, start_gui=True).run()