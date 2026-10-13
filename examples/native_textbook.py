"""参考资料数组到通用 member_loads 的显式输入转换，不求解、不修正结果。"""
from .textbook_reference import equivalent_model


def native_textbook_model(raw):
    model, _, _ = equivalent_model(raw)
    model['loads'] = []
    model['member_loads'] = []
    for load in raw['member_loads']:
        kind, plane, value = load['kind'], load['plane'], load['value']
        components = [0.,0.,0.]
        if kind in (1,2):
            components[1+plane] = value
            name = 'point_force' if kind == 1 else 'uniform_force'
        elif kind == 3:
            components[0] = value if plane == 0 else -value
            name = 'point_force' if plane == 0 else 'point_moment'
        elif kind == 6:
            # 原验证适配器所用附录分量：plane0 为 -My_RH，plane1 为 +Mz_RH。
            components[1+plane] = value*(1 if plane == 0 else -1)
            name = 'point_moment'
        else:
            raise ValueError('原题转换尚未登记该参考资料荷载类型')
        item = {'element_id': load['element'], 'kind': name,
                'coordinate_system': 'local', 'components': components}
        if kind == 2:
            item['interval'] = [0.,load['position']]
        else:
            item['position'] = load['position']
        model['member_loads'].append(item)
    model['section_queries'] = [
        {'id': i+1, 'element_id': item['element'], 'location': 'position',
         'position': item['position'], 'side': 'right'} for i,item in enumerate(raw['sections'])]
    return model
