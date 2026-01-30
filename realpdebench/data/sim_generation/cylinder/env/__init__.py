from gym.envs.registration import register

register(
    id='lilypad-v1',
    entry_point="env.flow_field_env:env"
)