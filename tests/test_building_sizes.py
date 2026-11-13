from rescue_env.envs.single_agent import RescueEnv

env = RescueEnv()

for seed in range(10):
    obs, info = env.reset(seed=seed)

    print(
        f"Seed {seed}: "
        f"Building={env.sim.h}x{env.sim.w}, "
        f"Memory={obs['memory'].shape}, "
        f"Valid={env.observation_space.contains(obs)}"
    )

env.close()

