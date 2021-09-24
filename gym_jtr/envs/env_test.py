from jtr_env import JtrEnv
from jtr_env_v1 import JtrEnvV1
from jtr_env_v2 import JtrEnvV2
from jtr_env_her import JtrEnvHER
from jtr_env_modelless import JtrEnvModelless
from jtr_env_modelless_v1 import JtrEnvModellessV1
from jtr_env_modelless_v2 import JtrEnvModellessV2
from jtr_env_modelless_v3 import JtrEnvModellessV3
from jtr_env_modelless_v4 import JtrEnvModellessV4
from jtr_env_suggested_v0 import JtrEnvSuggestedV0
from stable_baselines3.common.env_checker import check_env

env = JtrEnvSuggestedV0()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)