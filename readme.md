My project is about using camera intel realsense d455, in order to see the coordinates of the object on the table. My object is toy car, which is on the table. On the table, i also have a Kuka kr10 r1100 - 2 manipulator, which should to pick and place operation with my car object. Manipulator should move by kinematics. So, my camera is located above the table on the metal stick, in order to see table from high. So, my camera see the object and calculate the coordinate of the object, and send it to the manipulator, by ros2. In order to my manipulator pick object and then, put it to the already defined object.
The end effector of kuka manipulator is magnetic end effector, which work simple when end effector looks perpendicular down to the toy car, it magnets it, and then lift it, and put it to the already perdifined place. 



My project is in Isaac Sim

/home/rassul_pc/mono_depth_kuka/3d_model/arlan_project_srtand_usd_updated.usdz there are my table and camera holder

/home/rassul_pc/mono_depth_kuka/3d_model/car_object_arlan_usd.usdz this is my object toy car

/home/rassul_pc/mono_depth_kuka/3d_model/KR10_R1100_2_updated_description this is my kuka manipulator 



1) Kuka manipulator have to pick and place operation with car object, and put it to particular predefined place on the table. Car have to change its locate on the table each time.

2) Then, you need to create dataset of 1000 pictures of depth and rgb, in order to in the future i could train my own computer vision model. In each picture toy car should be in different locations on the table.

i have already put kuka manipulator to its proper location. and the same with table and realsense camera. The toy car is too on the table, but you will change it location on the table.