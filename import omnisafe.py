import omnisafe


env_id = 'SafetyCarGoal1-v0'

agent = omnisafe.Agent('TRPO', env_id)
agent.learn() 