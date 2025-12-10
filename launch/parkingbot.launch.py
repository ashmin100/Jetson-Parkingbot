from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="parkingbot",
                executable="parkingbot_node",
                name="parkingbot",
                output="screen",
                # 필요 시 아래 arguments에 --output 스트리밍 URI 추가 가능
            )
        ]
    )
