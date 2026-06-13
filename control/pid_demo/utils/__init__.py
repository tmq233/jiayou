from .plot_setup import setup_chinese_fonts
setup_chinese_fonts()

from .pid_controllers import (
    PIDPositional,
    PIDIncremental,
    PIDAntiWindupClamping,
    PIDDeadzone,
)
from .simulation import (
    simulate_car,
    simulate_car_with_backlash,
    cruise_metrics,
    DEFAULT_CAR,
)
