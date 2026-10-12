from .analysis import AnalysisResult, solve_space_frame
from .equations import SolverError
from .parameters import MaterialSectionInput, calculate_material_section
from .specialized import solve_plane_truss, solve_space_truss, solve_plane_frame


def solve_structure(model):
    from src.model import parse_model
    model=parse_model(model)
    kind=getattr(model,'structure_type','space_frame')
    return {'space_frame':solve_space_frame,'plane_frame':solve_plane_frame,
            'plane_truss':solve_plane_truss,'space_truss':solve_space_truss}[kind](model)

__all__ = ['AnalysisResult', 'SolverError', 'solve_space_frame',
           'MaterialSectionInput', 'calculate_material_section','solve_plane_frame','solve_plane_truss','solve_space_truss','solve_structure']
