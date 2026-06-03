# class_mapping.py
"""
UCMerced 21类到Potsdam/Toronto 4类的映射关系
"""

# UCMerced 21个类别
UCMERCED_CLASSES = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
    'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
    'storagetanks', 'tenniscourt'
]

# 映射关系
UCMERCED_TO_TARGET_MAPPING = {
    # 城市场景 (urban)
    'buildings': 'urban',
    'denseresidential': 'urban',
    'mediumresidential': 'urban',
    'sparseresidential': 'urban',
    'chaparral': 'urban',
    'parkinglot': 'urban',
    'mobilehomepark': 'urban',
    'overpass': 'urban',
    'intersection': 'urban',
    'freeway': 'urban',
    'airplane': 'urban',
    'baseballdiamond': 'urban',
    'tenniscourt': 'urban',
    'storagetanks': 'urban',
    'runway': 'urban',

    # 农田场景 (farmland)
    'agricultural': 'farmland',
    'golfcourse': 'farmland',

    # 山区场景 (mountain)
    'forest': 'mountain',

    # 水体场景 (water)
    'river': 'water',
    'harbor': 'water',
    'beach': 'water'
}

# 目标数据集4个类别
TARGET_CLASSES = ['urban', 'farmland', 'mountain', 'water']


def map_ucmerced_to_target(ucmerced_class):
    """将UCMerced类别映射到目标类别"""
    return UCMERCED_TO_TARGET_MAPPING.get(ucmerced_class, 'urban')  # 默认映射到urban


def get_class_weights():
    """获取类别权重，用于处理类别不平衡"""
    return {
        'urban': 1.0,
        'farmland': 1.2,
        'mountain': 1.5,
        'water': 1.8
    }