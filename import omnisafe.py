import omnisafe
import yaml
from omnisafe.utils.config import Config

def new_init(self, algo_name, env_id, config_overrides=None, *args, **kwargs):
    original_init(self, algo_name, env_id, *args, **kwargs)
    
    if config_overrides is not None:
        # 将Config对象转换为字典
        config_dict = self.cfgs.todict()
        
        # 更新字典中的值
        config_dict.update(config_overrides)
        
        # 将更新后的字典转换回Config对象
        self.cfgs = Config.dict2config(config_dict)
        
        print(f"Updated config: lagrangian_multiplier_init = {self.cfgs.lagrange_cfgs.lagrangian_multiplier_init}")

original_init = omnisafe.Agent.__init__
omnisafe.Agent.__init__ = new_init

env_id = 'SafetyPointGoal2-v0'
config_path = r"C:\Users\28906\Desktop\cl8\omnisafe-main\omnisafe\configs\on-policy\TRPOLag.yaml"
lagrangian_multiplier_init_values = [0.25, 0.5, 0.75]

for lagrangian_multiplier_init in lagrangian_multiplier_init_values:
    for seed in [0]:
        # 加载原始的YAML文件
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # 修改lagrangian_multiplier_init和seed的值
        config_dict['defaults']['lagrange_cfgs']['lagrangian_multiplier_init'] = lagrangian_multiplier_init
        config_dict['defaults']['seed'] = seed
        
        # 将修改后的配置写回原始的YAML文件
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file)
        
        # 将修改后的配置转换为Config对象
        config = Config.dict2config(config_dict)
        
        # 验证lagrangian_multiplier_init和seed的值是否已经被成功修改
        print(f"Expected lagrangian_multiplier_init value: {lagrangian_multiplier_init}")
        print(f"Actual lagrangian_multiplier_init value: {config.defaults.lagrange_cfgs.lagrangian_multiplier_init}")
        print(f"Expected seed value: {seed}")
        print(f"Actual seed value: {config.defaults.seed}")
        print()
        
        agent = omnisafe.Agent('TRPOLag', env_id)
        
        # 训练模型
        agent.learn()