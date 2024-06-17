# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of collecting offline data with OmniSafe."""

from omnisafe.common.offline.data_collector import OfflineDataCollector

env_name = 'SafetyPointGoal2-v0'
size = 110000
lamb = '1.75'  # 定义一个变量替代所有的"0.25"
base_path = r"C:\Users\28906\Desktop\line\data\2wX1000\TRPOLag-{SafetyPointGoal2-v0}"
steps_per_epoch = 10000  # 每个epoch读取的步数,可从外部输入

agents = []
for epoch in range(0, 1001, 100):  # 从0到1000,每隔10个epoch取一个
    agent_path = fr"{base_path}\{lamb}\torch_save"
    model_name = fr"{base_path}\{lamb}\torch_save\epoch-{epoch}.pt"
    agents.append((agent_path, model_name, steps_per_epoch))

save_dir = './data'

if __name__ == '__main__':
    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)