import os
from glob import glob
from setuptools import setup

package_name = 'ev3_pi'

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
            "charlie_node = ev3_pi.charlie_node",
            "compass_node = ev3_pi.compass_node",
            "ultrasonic_distance_node = ev3_pi.ultrasonic_distance_node",
            "object_detector_node = ev3_pi.object_detector_node",
            "person_follower_node = ev3_pi.person_follower_node"
        ],
    },
)
