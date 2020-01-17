from eco_sim import Environment # , start_with_gui

import torch
import torch.nn.functional as fun
import numpy
import random
import time
import math

device=torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device=torch.device("cuda")

def transposed_tensor(arr):
    # (map_height, map_width, max_reps_per_square : D, rep_size : C) => (N, C, H, W, D)
    return torch.from_numpy(numpy.transpose(arr)).to(device)


def blowup(t0, t1):
    out = torch.zeros((t0.shape[0],) + t1.shape)
    for i in range(0, t0.shape[1] ):
        for j in range(0, t0.shape[1]):
            out[i, j, :] = t0[i, j] * t1[j, :]
    return out

class ActionValues(torch.nn.Module):
    def __init__(
            self,
            action_space_size=Environment.action_space_size(),
            map_height=Environment.map_height(),
            map_width=Environment.map_width(),
            max_reps_per_square=Environment.max_reps_per_square(),
            rep_size=Environment.rep_size(),
            phys_size=Environment.physical_rep_size(),
            mental_size=Environment.mental_rep_size(),
            max_other_agents=12):
        super(ActionValues, self).__init__()
        self.h = map_height
        self.w = map_width
        self.d = max_reps_per_square
        self.c0 = rep_size
        self.max_other_agents = max_other_agents
        self.phys_size = phys_size
        self.mental_size = mental_size
        # stack (map_height: D, map_width : W, max_reps_per_square : H, rep_size : C)T => (N, C, H, W, D)
        # kernel (h, w, d)
        kernel = (max_reps_per_square, 1, 1)
        self.c1 = 16
        self.c2 = 16
        self.c3 = 8
        self.cp = phys_size
        self.cm = mental_size
        hsz = self.c3 * map_height * map_width


        # Preprocessing for the map:

        self.conv1 = torch.nn.Conv3d(rep_size, self.c1, groups=1, kernel_size=kernel)
        #(N, c1, 1, W, D) => (N, c1, W, D)
        self.bn1 = torch.nn.BatchNorm3d(self.c1)
        self.conv2 = torch.nn.Conv2d(self.c1, self.c2, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.c2)
        self.conv3 = torch.nn.Conv2d(self.c2, self.c3, kernel_size=(5, 5), stride=1, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(self.c3)


        ag_size = phys_size + mental_size

        # Preprocessing for the physical / mental state of one entity, used for both the own state and that of others (if visible)
        self.phys = torch.nn.Linear(phys_size, phys_size)
        self.ment = torch.nn.Linear(mental_size, mental_size)

        # Overall state representation with memory

        self.state = None
        self.lstm = torch.nn.LSTM(self.c3 * self.h * self.w + ag_size + max_other_agents * self.cp + max_other_agents * self.cm, hsz)


        self.old_ment = None

        # Head for translating to action values
        self.act_hidden = torch.nn.Linear(hsz, 128)
        self.act = torch.nn.Linear(128, action_space_size)

        # Head for anticipating mental states after a sim step
        self.infer_mental = torch.nn.Linear(hsz, max_other_agents * mental_size)

    # x: map batch, p: all physical states, m: all mental states, vis: mutual visibility,
    # mix_ratio: how much of the mental state input for others an active entity sees is based on the actual mental state,
    # as opposed to its own previous estimate.
    def forward(self, x, p, m, vis, mix_ratio=0.9):
        k = x.shape[0]
        x = fun.relu(self.bn1(self.conv1(x)))
        x = torch.squeeze(x, dim=2)
        x = fun.relu(self.bn2(self.conv2(x)))
        x = fun.relu(self.bn3(self.conv3(x)))
        ag = torch.cat((fun.relu(self.phys(p)), fun.relu(self.ment(m))), 1)

        # Process all physical states once as batch (zero for any potential entity slots that are empty),
        # then repeat as input for each active entity, setting to zero when there is no visibility.
        phys = torch.zeros((self.max_other_agents, self.phys_size))
        phys[:k, :] = p
        other_p = fun.relu(self.phys(phys))
        phys_inp = blowup(vis, other_p)

        # First create appropriate input mix between previous estimate and objective input, then process.
        ment = torch.zeros((1, self.max_other_agents, self.mental_size))
        ment[0, :k, :] = m
        other_m = mix_ratio * ment.expand(k, -1, -1)
        if self.old_ment is not None:
            other_m += (1.0 - mix_ratio) * self.old_ment
        ment_inp = fun.relu(self.ment(other_m.view(k * self.max_other_agents, -1)))

        x, s = self.lstm(
            torch.cat((x.view(1, k, -1), ag.view(1, k, -1),
            phys_inp.view(1, k, -1),
            ment_inp.view(1, k, -1)
            ), 2), self.state)
        self.state = s
        x = fun.relu(x.view(k, -1))
        self.old_ment = self.infer_mental(x).view(k, self.max_other_agents, -1)

        x = fun.relu(self.act_hidden(x))
        act = self.act(x)

        return act, self.old_ment

    def reset_state(self):
        self.state = None
        self.old_ment = None

    def detach_state(self):
        self.state = (self.state[0].detach(), self.state[1].detach())
        self.old_ment = self.old_ment.detach()

    def remap(self, remapping, k):
        after = 0
        for (i, new_i) in remapping:
            if new_i is not None:
                after = new_i + 1
                self.state[0][:, new_i, :] = self.state[0][:, i, :]
                self.state[1][:, new_i, :] = self.state[1][:, i, :]
                self.old_ment[new_i, :, :] = self.old_ment[i, :, :]
        for (i, new_i) in remapping:
            if new_i is not None:
                self.old_ment[:after, new_i, :] = self.old_ment[:after, i, :]
        if len(remapping) > 0 and k > after:
            s0 = self.state[0].shape
            s1 = self.state[1].shape
            sm = self.old_ment.shape
            self.state[0][:, after:k, :] = torch.zeros((s0[0], k - after, s0[2]))
            self.state[1][:, after:k, :] = torch.zeros((s0[0], k - after, s1[2]))
            self.old_ment[after:, :, :] = torch.zeros((k - after, sm[1], sm[2]))
            self.old_ment[:, after:, :] = torch.zeros((k, self.max_other_agents - after, sm[2]))


class QLearner:
    def __init__(self, env=None, seed=0, agent_count=8, pretrained=None, start_gui=False):
        self.agent_count = agent_count

        self.env = env if env is not None and isinstance(env, Environment) else Environment(seed)
        if start_gui:
            self.env.start_gui()
        self.next_seed = seed + 1
        if pretrained is not None and isinstance(pretrained, ActionValues):
            print("pretrained")
            self.policy = pretrained
        else:
            print("new")
            self.policy = ActionValues()
        self.value = ActionValues().eval()
        self.optimizer = torch.optim.RMSprop(self.policy.parameters())
        self.optimizer

    def run(self, epochs=400, wait=0):
        time.sleep(wait)
        action_losses = []
        mental_losses = []
        running_action_loss = 0.0
        running_mental_loss = 0.0
        total_count = 0
        action_space = Environment.action_space_size()
        for epoch in range(1, epochs):

            self.policy.reset_state()
            self.value.reset_state()
            current_count = 0
            in_epoch_count = 0
            self.env.reset(self.next_seed)
            self.next_seed += 1
            registered = self.env.register_agents(self.agent_count)
            k = len(registered)
            obsv, rewards, suggested_actions, physical, mental, visibility, remappings, complete  = self.env.state()
            vis = torch.tensor(visibility)
            rewards = torch.tensor(rewards).view(-1, 1)
            done = complete
            rand = random.Random()
            inp = torch.stack([transposed_tensor(np) for np in obsv])
            men = torch.from_numpy(mental).to(device)
            phys = torch.from_numpy(physical).to(device)
            death_count = 0

            # Weight of real mental states of other agents relative to previously estimated mental states in input
            # for next estimate.

            mix_ratio = math.exp(epoch / -10.0)

            # Probability of choosing the action suggested by the hardcoded behavior, choosing randomly,
            # or choosing the action with maximal expected value.

            p_choose_suggested = 0.5 * math.exp(epoch / -200.0)
            p_choose_random = 0.5 * math.exp(epoch / -10.0)
            p_choose_max = 1.0 - p_choose_random - p_choose_suggested

            # Each agent sticks with one decision procedure for a while so that mid term dependencies,
            # particularly in the suggested actions, can be better observed.

            choose = rand.choices([0, 1, 2], weights=[p_choose_max, p_choose_suggested, p_choose_random], k=k)
            while not done and in_epoch_count < 1000:
                if death_count > 0:
                    choose = rand.choices([0, 1, 2], weights=[p_choose_max, p_choose_suggested, p_choose_random], k=k)
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
                pol, ment_inf = self.policy.forward(inp, phys, men, vis, mix_ratio=mix_ratio)
                actions = []
                for i in range(0, k):
                    choice = choose[i]
                    if choice == 0:
                        actions.append(torch.argmax(pol[i]))
                    elif choice == 1:
                        actions.append(suggested_actions[i])
                    elif choice == 2:
                        actions.append(rand.choice(range(0, action_space)))
                obsv, new_rewards, new_suggested_actions, physical, mental, visibility, remappings, complete = self.env.step(actions)
                vis = torch.tensor(visibility)
                done = complete
                inp = torch.stack([transposed_tensor(np) for np in obsv])
                men = torch.from_numpy(mental).to(device)
                phys = torch.from_numpy(physical).to(device)
                new_rewards = torch.tensor(new_rewards).view(-1, 1)
                expected_rewards = new_rewards - rewards
                death_count = 0
                with torch.no_grad():
                    m,_ = self.value.forward(inp, phys, men, vis, mix_ratio=mix_ratio)[0].max(1, keepdim=True)
                    for (i, target) in remappings:
                        if target is None:
                            death_count += 1
                            m[i, 0] = 0.0
                    expected_rewards += m
                rewards = new_rewards
                action_loss = fun.smooth_l1_loss(pol.gather(1, torch.tensor(actions).view(k, 1)), expected_rewards)
                ment_loss = fun.smooth_l1_loss(ment_inf[:, :k, :], men.expand(k, -1, -1))
                loss = action_loss + ment_loss
                running_action_loss += action_loss.item()
                running_mental_loss += ment_loss.item()
                if done or current_count > 20:
                    self.value.load_state_dict(self.policy.state_dict())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    avg_mental_loss = running_mental_loss / current_count
                    avg_action_loss = running_action_loss / current_count
                    print(avg_action_loss, avg_mental_loss)
                    print(rewards.view(-1))
                    action_losses.append(avg_action_loss)
                    mental_losses.append(avg_mental_loss)
                    current_count = 0
                    running_action_loss = 0.0
                    running_mental_loss = 0.0
                    self.policy.detach_state()
                    choose = rand.choices([0, 1, 2], weights=[p_choose_max, p_choose_suggested, p_choose_random], k=k)
                else:
                    loss.backward(retain_graph=True)
            torch.save(self.policy, "qlearner.bak")




if __name__ == '__main__':
    # start_gui=True, pretrained=torch.load("qlearner.bak")
    QLearner(0, start_gui=True).run()