"""用户显式选用教材规则时生成局部轴，不能用于猜测图片截面朝向。

正文 p119～120 式5-6/5-10；附录IV p183/p195～196。
两处的零转角参考轴不同，必须显式区分。
"""
import numpy as np


def textbook_basis(xi, xj, roll=0., style='chapter5'):
    """返回局部到总体的 3×3 基 Q；roll 为明确给定的截面转角(rad)。"""
    xi, xj = np.asarray(xi,dtype=float), np.asarray(xj,dtype=float)
    if xi.shape != (3,) or xj.shape != (3,) or not np.all(np.isfinite([xi,xj])):
        raise ValueError('教材轴函数需要有限三维坐标')
    if style not in ('chapter5','appendix4') or not np.isfinite(roll):
        raise ValueError('教材轴类型或截面转角无效')
    delta = xj-xi
    length = np.linalg.norm(delta)
    if not np.isfinite(length) or length <= 0:
        raise ValueError('教材轴函数遇到零长度或无效长度')
    ex = delta/length
    horizontal = np.hypot(ex[0],ex[1])
    if horizontal <= 1e-12:
        # 正文与附录的竖杆特例都取基准 y=总体 Y。
        ey0 = np.array([0.,1.,0.])
        ey0 -= ex*np.dot(ex,ey0)
        ey0 /= np.linalg.norm(ey0)
    else:
        # 正文式5-6选 dn<0；附录IV的零角 y 轴选正 Z 投影。
        ey0 = np.array([ex[2]*ex[0]/horizontal, ex[2]*ex[1]/horizontal, -horizontal])
        if style == 'appendix4':
            ey0 *= -1
    ez0 = np.cross(ex,ey0)
    if style == 'appendix4' and horizontal <= 1e-12 and ex[2] > 0:
        # 附录IV-3 的竖直正向杆特殊式 t1，不能从一般杆的分母形式直接取极限。
        ey = np.cos(roll)*ey0 - np.sin(roll)*ez0
    else:
        # 正文式5-10，以及附录IV-3的一般杆 t3 / 竖直负向杆 t2。
        ey = np.cos(roll)*ey0 + np.sin(roll)*ez0
    return np.column_stack((ex,ey,np.cross(ex,ey)))
