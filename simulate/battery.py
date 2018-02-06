import os

from pathlib import Path
import pandas as pd


class Battery(object):
    """
        - initial_battery_state_of_charge is the initial state of charge of the
          battery (0.0 in our examples)
        - battery_capacity is the battery capacity in Wh
        - battery_charging_power_limit in W
        - battery_discharging_power_limit in W (counted negatively)
        - battery_charging_efficiency
        - battery_discharging_efficiency
    """
    def __init__(self,
                 current_charge=0.0,
                 capacity=0.0,
                 charging_power_limit=1.0,
                 discharging_power_limit=-1.0,
                 charging_efficiency=0.95,
                 discharging_efficiency=0.95):
        self.current_charge = current_charge
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
