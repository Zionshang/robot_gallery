cmake_minimum_required(VERSION 3.8)
project(x5_description)

find_package(ament_cmake REQUIRED)

install(
		DIRECTORY launch meshes urdf xml
		DESTINATION share/${PROJECT_NAME}/
)

ament_package()
