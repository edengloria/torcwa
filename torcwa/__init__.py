from .torch_eig import Eig
from .geometry import geometry, rcwa_geo
from .rcwa import rcwa
from .api import Layer, Output, PlaneWave, RCWA, Stack, UnitCell
from .materials import MaterialGrid
from . import materials as material
from . import v2
from . import core

geometry.Grid = geometry

__author__ = '''Changhyun Kim'''
__version__ = '0.3.0.dev0'

__all__ = [
    'Eig',
    'Layer',
    'MaterialGrid',
    'Output',
    'PlaneWave',
    'RCWA',
    'Stack',
    'UnitCell',
    'core',
    'geometry',
    'material',
    'rcwa_geo',
    'rcwa',
    'v2',
]
