from typing import TYPE_CHECKING

import numpy as np

# Types
if TYPE_CHECKING:
    from typing import TypeAlias
    from collections.abc import Callable
    from typing import Any
    from .._aux import ODEA_return
    from .._aux import npAFloat64

    ODEA2Type: TypeAlias = Callable[[np.float64, npAFloat64, npAFloat64, Any],
                                    ODEA_return]
else:
    ODEA2Type = None
