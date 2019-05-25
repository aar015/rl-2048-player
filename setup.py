from setuptools import setup


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(name='rl2048player',
      version='1.0',
      description='Reinforcement Learning Agent capble of playing 2048',
      long_description=readme(),
      url='https://github.com/aar015/rl-2048-player',
      author='aar015',
      packages=['rl2048player'],
      install_requires=[
          'imageio',
          'matplotlib',
          'numpy'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.6'
      ],
      keywords='2048 RL',
      include_package_data=True,
      zip_safe=False)
