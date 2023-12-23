import os
from glob import glob
from setuptools import setup

package_name = 'lego'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "thomas_node = lego.thomas_node",
            "compass_node = lego.compass_node",
            "ultrasonic_distance_node = lego.ultrasonic_distance_node",
            "object_detector_node = lego.object_detector_node",
            "person_follower_node = lego.person_follower_node"
        ],
    },
)
