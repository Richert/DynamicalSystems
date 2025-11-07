from rectipy import Network
from pyrates import OperatorTemplate, NodeTemplate, EdgeTemplate, CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt

op = OperatorTemplate(
    name="relu",
    equations=[""]
)