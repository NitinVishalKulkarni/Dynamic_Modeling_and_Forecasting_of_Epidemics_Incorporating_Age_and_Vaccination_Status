from setuptools import setup
from gym.envs.registration import register


setup(
    name='seihrd',
    version='0.0.0',
    packages=['seihrd'],
    package_dir={'': 'src'},
    install_requires=['gymnasium'],
)

register(
    id='seihrd-v0',
    entry_point='seihrd.seihrd_env:SeihrdEnv',
)
