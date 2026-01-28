"""EBT film calibration classes and predefined calibration curves."""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Calibration:
    """EBT3 calibration: dose = a0 + a1*netOD + a2*netOD^2 + a3*netOD^3
    
    Parameters
    ----------
    a0 : float
        Constant term (offset)
    a1 : float
        Linear coefficient
    a2 : float
        Quadratic coefficient
    a3 : float
        Cubic coefficient
        
    Examples
    --------
    >>> calib = Calibration(a1=9.62189, a3=78.75125)
    >>> calib(0.5)  # Calculate dose for netOD=0.5
    """
    a0: float = 0
    a1: float = 0
    a2: float = 0
    a3: float = 0

    def __call__(self, x: npt.ArrayLike) -> npt.NDArray:
        """Calculate dose from net optical density.
        
        Parameters
        ----------
        x : array_like
            Net optical density value(s)
            
        Returns
        -------
        np.ndarray
            Dose in Gy
        """
        x = np.asarray(x)
        return self.a0 + self.a1*x + self.a2*x**2 + self.a3*x**3

    def __repr__(self) -> str:
        terms = []
        if self.a0 != 0:
            terms.append(f'{self.a0}')
        if self.a1 != 0:
            terms.append(f'{self.a1}*x')
        if self.a2 != 0:
            terms.append(f'{self.a2}*x^2')
        if self.a3 != 0:
            terms.append(f'{self.a3}*x^3')
        return 'f(x) = ' + ' + '.join(terms) if terms else 'f(x) = 0'


# Predefined calibration curves
ebt3_proton_calib_20Gy = Calibration(a1=9.62189, a3=78.75125)
"""EBT3 proton calibration for 20 Gy dose range."""
