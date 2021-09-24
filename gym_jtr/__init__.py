from gym.envs.registration import register

register(
    id='jtr-v0',
    entry_point='gym_jtr.envs:JtrEnvV0',
)

register(
    id='jtr-v1',
    entry_point='gym_jtr.envs:JtrEnvV1',
)

register(
    id='jtr-v2',
    entry_point='gym_jtr.envs:JtrEnvV2',
)

register(
    id='jtr-her-v0',
    entry_point='gym_jtr.envs:JtrEnvHERV0',
)

register(
    id='jtr-modelless-v0',
    entry_point='gym_jtr.envs:JtrEnvModellessV0',
)

register(
    id='jtr-modelless-v1',
    entry_point='gym_jtr.envs:JtrEnvModellessV1',
)

register(
    id='jtr-modelless-v2',
    entry_point='gym_jtr.envs:JtrEnvModellessV2',
)

register(
    id='jtr-modelless-v3',
    entry_point='gym_jtr.envs:JtrEnvModellessV3',
)

register(
    id='jtr-modelless-v4',
    entry_point='gym_jtr.envs:JtrEnvModellessV4',
)

register(
    id='jtr-suggested-v0',
    entry_point='gym_jtr.envs:JtrEnvSuggestedV0',
)