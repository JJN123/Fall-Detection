Place Fall/NonFall frames here. The directory structure should be as follows:


if dset == 'Thermal':
    path_Fall = '/Thermal/Fall/Fall*'
    path_ADL =  '/Thermal/NonFall/ADL*'

elif dset == 'UR':
    path_Fall =  '/UR_Kinect/Fall/original/Fall*'
    path_ADL =  '/UR_Kinect/NonFall/original/adl*'

elif dset == 'UR-Filled':
    path_Fall =  '/UR_Kinect/Fall/filled/Fall*'
    path_ADL =  '/UR_Kinect/NonFall/filled/adl*'

elif dset == 'SDU':
    path_Fall =  '/SDUFall/Fall/Fall*/Depth'
    path_ADL =  '/SDUFall/NonFall/ADL*/Depth'

elif dset == 'SDU-Filled':
    path_Fall =  '/SDUFall/Fall/Fall*/Filled'
    path_ADL =  '/SDUFall/NonFall/ADL*/Filled'
