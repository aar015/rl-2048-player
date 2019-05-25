from setuptools import setup

setup(name='rl2048player',
      version='1.0',
      description='Reinforcement Learning Agent capble of playing 2048',
      url='https://github.com/aar015/rl-2048-player',
      author='aar015',
      packages=['rl2048player'],
      install_requires=[
          csv,
          cv2,
          imageio,
          matplotlib,
          numpy,
          pickle,
          random,
          abc
      ],
      zip_safe=False)
