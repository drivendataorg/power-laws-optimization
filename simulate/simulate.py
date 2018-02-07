import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from battery_controller import BatteryContoller
from battery import Battery


class Simulation(object):
    """ Handles running a simulation.
    """
    def __init__(self,
                 data,
                 battery,
                 actual_previous_load=0.0,
                 actual_previous_pv=0.0):
        """ Creates initial simulation state based on data passed in.

            :param data: contains all the time series needed over the considered period
            :param battery: is a battery instantiated with 0 charge and the relevant properties
            :param actual_previous_load: (optional) of the timestep right before the simulation starts (default=0)
            :param actual_previous_load: (optional) actual_previous_pv of the timestep right before the simulation
                starts (default=0)
        """

        self.data = data
        self.load_columns = data.columns.str.startswith('load_')
        self.pv_columns = data.columns.str.startswith('pv_')
        self.price_sell_columns = data.columns.str.startswith('price_sell_')
        self.price_buy_columns = data.columns.str.startswith('price_buy_')

        # initialize money at 0.0
        self.money_spent = 0.0
        self.money_spent_without_battery = 0.0

        # battery initialization
        self.battery = battery

        # building initialization
        self.actual_previous_load = actual_previous_load
        self.actual_previous_pv = actual_previous_pv

    def run(self):
        """ Executes the simulation by iterating through each of the data points
            It returns both the electricity cost spent using the battery and the
            cost that would have been incurred with no battery.
        """
        battery_controller = BatteryContoller()

        for current_time, timestep in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='timesteps'):
            self.simulate_timestep(battery_controller, current_time, timestep)

        return self.money_spent, self.money_spent_without_battery

    def simulate_timestep(self, battery_controller, current_time, timestep):
        """ Executes a single timestep using `battery_controller` to get
            a proposed state of charge and then calculating the cost of
            making those changes.

            :param battery_controller: The battery controller
            :param current_time: the timestamp of the current time step
            :param timestep: the data available at this timestep
        """
        # get proposed state of charge from the battery controller
        proposed_state_of_charge = battery_controller.propose_state_of_charge(
            current_time,
            self.battery,
            self.actual_previous_load,
            self.actual_previous_pv,
            timestep[self.price_buy_columns],
            timestep[self.price_sell_columns],
            timestep[self.load_columns],
            timestep[self.pv_columns]
        )

        # get energy required to achieve the proposed state of charge
        grid_energy, battery_energy_change = self.simulate_battery_charge(self.battery.current_charge,
                                                                          proposed_state_of_charge,
                                                                          timestep.actual_consumption,
                                                                          timestep.actual_pv)

        grid_energy_without_battery = timestep.actual_consumption - timestep.actual_pv

        # buy or sell energy depending on needs
        price = timestep.price_buy_00 if grid_energy >= 0 else timestep.price_sell_00
        price_without_battery = timestep.price_buy_00 if grid_energy_without_battery >= 0 else timestep.price_sell_00

        # calculate spending based on price per kWh and energy per Wh
        self.money_spent += grid_energy * (price / 1000.)
        self.money_spent_without_battery += grid_energy_without_battery * (price_without_battery / 1000.)

        # update current state of charge
        self.battery.current_charge += battery_energy_change / self.battery.capacity
        self.actual_previous_load = timestep.actual_consumption
        self.actual_previous_pv = timestep.actual_pv

    def simulate_battery_charge(self, initial_state_of_charge, proposed_state_of_charge, actual_consumption, actual_pv):
        """ Charges or discharges the battery based on what is desired and
            available energy from grid and pv.

            :param initial_state_of_charge: the current state of the battery
            :param proposed_state_of_charge: the proposed state for the battery
            :param actual_consumption: the actual energy consumed by the building
            :param actual_pv: the actual pv energy produced and available to the building
        """
        # charge is bounded by what is feasible
        proposed_state_of_charge = np.clip(proposed_state_of_charge, 0.0, 1.0)

        # calculate proposed energy change in the battery
        target_energy_change = (proposed_state_of_charge - initial_state_of_charge) * self.battery.capacity

        # efficiency can be different whether we intend to charge or discharge
        if target_energy_change >= 0:
            efficiency = self.battery.charging_efficiency
            target_charging_power = target_energy_change / ((15. / 60.) * efficiency)
        else:
            efficiency = self.battery.discharging_efficiency
            target_charging_power = target_energy_change * efficiency / (15. / 60.)

        # actual power is bounded by the properties of the battery
        actual_charging_power = np.clip(target_charging_power,
                                        self.battery.discharging_power_limit,
                                        self.battery.charging_power_limit)

        # actual energy change is based on the actual power possible and the efficiency
        if actual_charging_power >= 0:
            actual_energy_change = actual_charging_power * (15. / 60.) * efficiency
        else:
            actual_energy_change = actual_charging_power * (15. / 60.) / efficiency

        # what we need from the grid = (the power put into the battery + the consumption) - what is available from pv
        grid_energy = (actual_charging_power * (15. / 60.) + actual_consumption) - actual_pv

        # if positive, we are buying from the grid; if negative, we are selling
        return grid_energy, actual_energy_change


if __name__ == '__main__':
    simulation_dir = (Path(__file__)/os.pardir/os.pardir).resolve()
    data_dir = simulation_dir/'data'
    output_dir = simulation_dir/'output'

    # load available metadata to determine the runs
    metadata_path = data_dir/'metadata.csv'
    metadata = pd.read_csv(metadata_path, index_col=0)

    # store results of each run
    results = []

    # # execute two runs with each battery for every row in the metadata file:
    for site_id, parameters in tqdm(metadata.iterrows(), desc='sites', total=metadata.shape[0]):
        site_data_path = data_dir/"submit"/f"{site_id}.csv"

        if site_data_path.exists():
            site_data = pd.read_csv(site_data_path,
                                    parse_dates=['timestamp'],
                                    index_col='timestamp')

            for run_id in [1, 2]:
                # create the battery for this run
                batt = Battery(capacity=parameters[f"Battery_{run_id}_Capacity"],
                               charging_power_limit=parameters[f"Battery_{run_id}_Power"],
                               discharging_power_limit=-parameters[f"Battery_{run_id}_Power"],
                               charging_efficiency=parameters[f"Battery_{run_id}_Charge_Efficiency"],
                               discharging_efficiency=parameters[f"Battery_{run_id}_Discharge_Efficiency"])

                # execute the simulation
                sim = Simulation(site_data, batt)
                money_spent, money_no_batt = sim.run()

                # store the results
                results.append({
                    'run_id': f"{site_id}_{run_id}",
                    'site_id': site_id,
                    'battery_id': run_id,
                    'money_spent': money_spent,
                    'money_no_batt': money_no_batt,
                    'score': money_spent / money_no_batt,
                })

    # write all results out to a file
    results_df = pd.DataFrame(results).set_index('run_id')
    results_df.to_csv(output_dir/'results.csv')
