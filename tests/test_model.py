import copy
import unittest

from pydantic import ValidationError
from src.model import StructuralModel
from src.solver import solve_space_frame
from .helpers import cantilever_data


class ModelTests(unittest.TestCase):
    def test_noncontinuous_ids(self):
        model = StructuralModel.model_validate(cantilever_data())
        self.assertEqual([n.id for n in model.nodes], [10, 90])

    def test_invalid_inputs(self):
        changes = [
            lambda d: d['nodes'][1].update(id=10),
            lambda d: d['elements'][0].update(node_i=999),
            lambda d: d['elements'][0].update(node_j=10),
            lambda d: d['elements'][0].update(material_id=999),
            lambda d: d['elements'][0].update(section_id=999),
            lambda d: d['sections'][0].update(A=0),
            lambda d: d['sections'][0].update(J=-1),
            lambda d: d['materials'][0].update(E=float('inf')),
            lambda d: d['nodes'][0].update(coordinates=[float('nan'), 0., 0.]),
            lambda d: d['nodes'][0].update(restrained=[True]*5),
            lambda d: d['loads'][0].update(components=[1., 2.]),
            lambda d: d['loads'][0].update(node_id=999),
            lambda d: d['elements'][0].update(local_y=[1., 0., 0.]),
            lambda d: d['elements'][0].pop('local_y'),
            lambda d: d['elements'][0].update(local_y=[0., 0., 0.]),
            lambda d: d.update(units='mm'),
            lambda d: d.update(convention='right_hand'),
            lambda d: d['elements'][0].update(release=True),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                data = copy.deepcopy(cantilever_data())
                change(data)
                with self.assertRaises(ValidationError):
                    StructuralModel.model_validate(data)

    def test_mutated_instance_revalidated(self):
        model = StructuralModel.model_validate(cantilever_data())
        model.elements[0].node_j = 999
        with self.assertRaises(ValidationError):
            solve_space_frame(model)
