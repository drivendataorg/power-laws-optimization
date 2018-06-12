"""
This module was created merging different modules into a single one.
This was a requirement of the challenge.

For merging the modules the notebook 041_prepare_submission was used.
For computing the coeficients that can be found on the assets folder the
notebook 039_forecast_accuracy_optimization_v6_keras_just_past_data
was used.
"""
from datetime import datetime
from functools import lru_cache
import json
import numpy as np
import pandas as pd

################################################################################
#  battery_controller_base.py
################################################################################



class BatteryContoller(object):
    """ The BatteryContoller class handles providing a new "target state of charge"
        at each time step.
        This class worries more about optimizing the cost than ortography.

        This class is instantiated by the simulation script, and it can
        be used to store any state that is needed for the call to
        propose_state_of_charge that happens in the simulation.

        The propose_state_of_charge method returns the state of
        charge between 0.0 and 1.0 to be attained at the end of the coming
        quarter, i.e., at time t+15 minutes.

        The arguments to propose_state_of_charge are as follows:
        :param site_id: The current site (building) id in case the model does different work per site
        :param timestamp: The current timestamp inlcuding time of day and date
        :param battery: The battery (see battery.py for useful properties, including current_charge and capacity)
        :param actual_previous_load: The actual load of the previous quarter.
        :param actual_previous_pv_production: The actual PV production of the previous quarter.
        :param price_buy: The price at which electricity can be bought from the grid for the
          next 96 quarters (i.e., an array of 96 values).
        :param price_sell: The price at which electricity can be sold to the grid for the
          next 96 quarters (i.e., an array of 96 values).
        :param load_forecast: The forecast of the load (consumption) established at time t for the next 96
          quarters (i.e., an array of 96 values).
        :param pv_forecast: The forecast of the PV production established at time t for the next
          96 quarters (i.e., an array of 96 values).

        :returns: proposed state of charge, a float between 0 (empty) and 1 (full).
    """
    def __init__(self):
        self.epoch = 0
        self.optimizer = None
        self.timestep_idx = 0

        self.before_previous_balance = None
        self.coef = None

        self.subsample = 1 #This controls the speed. The higher the number the faster
        self.PERIOD_DURATION = 960
        self.FORECAST_LENGTH = 96

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):
        # Compute the features for creating a forecast
        timestep_balance = load_forecast.values[0] - pv_forecast.values[0]
        previous_balance = actual_previous_load - actual_previous_pv_production
        current_balance = self._get_current_balance_prediction(
            timestep_balance, previous_balance, site_id, timestamp)

        if self.optimizer is None or self.timestep_idx >= self.subsample:
            self.optimizer = self._get_optimizer(price_sell, price_buy, pv_forecast,
                                                 load_forecast, current_balance, battery)
            self.timestep_idx = 0


        next_state = self.optimizer.get_fine_grain_policy_for_timestep(
            self.timestep_idx, current_balance, battery.current_charge)

        if next_state is None:
            self.optimizer = self._get_optimizer(price_sell, price_buy, pv_forecast,
                                                 load_forecast, current_balance, battery)
            self.timestep_idx = 0
            next_state = self.optimizer.get_fine_grain_policy_for_timestep(
                self.timestep_idx, current_balance, battery.current_charge)

        self.timestep_idx += 1
        self.epoch += 1
        return next_state

    def _prepare_data_for_optimizer(self, price_sell, price_buy, pv_forecast,
                                    load_forecast, epochs_to_end, current_balance):
        """
        Creates a period object for later optimizing, replace the first value
        of the balance using the given prediction
        """
        price_sell = price_sell.values
        price_buy = price_buy.values
        load = load_forecast.values
        pv = pv_forecast.values
        balance = load - pv
        balance[0] = current_balance

        if epochs_to_end < self.FORECAST_LENGTH:
            price_sell = price_sell[:epochs_to_end]
            price_buy = price_buy[:epochs_to_end]
            load = load[:epochs_to_end]
            pv = pv[:epochs_to_end]
            balance = balance[:epochs_to_end]
        period = Period(price_sell, price_buy, load, pv, balance)
        return period

    def _get_optimizer(self, price_sell, price_buy, pv_forecast,
                       load_forecast, current_balance, battery):
        """
        Creates a period object with the forecast and feeds it to the optimizer.
        Returns the optimizer object after doing the optimization.

        When less than a day remains to end the period it crops the forecast
        and allows a bigger degree of freedom.
        """
        epochs_to_end = self.PERIOD_DURATION - self.epoch
        period = self._prepare_data_for_optimizer(price_sell, price_buy, pv_forecast,
                                                  load_forecast, epochs_to_end, current_balance)
        if epochs_to_end < self.FORECAST_LENGTH:
            optimizer = PeriodOptimizer(period, battery, allow_end_optimization=True)
        else:
            optimizer = PeriodOptimizer(period, battery, allow_end_optimization=False)
        optimizer.optimize()
        return optimizer

    def _get_current_balance_prediction(self, timestep_balance,
                                        previous_balance, site_id, timestamp):
        """
        Returns a prediction for the balance of the system

        It uses:
        * The forecast of the balance
        * The actual balance of the previous timestep
        * The actual balance of the before previous timestep

        The coefficients have been computed using keras using past data of the
        same site
        """
        if self.before_previous_balance is None:
            self.before_previous_balance = previous_balance
            return timestep_balance

        if self.coef is None:
            self.coef = self._get_coefs(site_id, timestamp)

        current_balance = timestep_balance*self.coef[0] \
                        + previous_balance*self.coef[1] \
                        + self.before_previous_balance*self.coef[2]

        self.before_previous_balance = previous_balance
        return current_balance

    @staticmethod
    def _get_coefs(site_id, timestamp):
        with open('simulate/assets/coefs.json', 'r') as f:
            coef_dict = json.load(f)
        coef_dict = coef_dict['%i' % site_id]

        timestamp_options = list(coef_dict.keys())
        selected_timestamp = _get_selected_timestamp(str(timestamp), timestamp_options)
        if selected_timestamp is None:
            return [1, 0, 0]
        else:
            return coef_dict[selected_timestamp]



def _get_datetime(text_date):
    return datetime.strptime(text_date, "%Y-%m-%d %H:%M:%S")

def _get_selected_timestamp(ref_timestamp, timestamp_options):
    """Finds the closest past timestamp. If there are not candidates returns None"""
    ref_date = _get_datetime(ref_timestamp)
    time_dist = [(ref_date - _get_datetime(_timestamp)).total_seconds() \
                    for _timestamp in timestamp_options]

    possible_idx = [i for i, _ in enumerate(time_dist) if time_dist[i] > 0]

    if not possible_idx:
        return None
    time_dist = [time_dist[i] for i in possible_idx]
    selected_idx = possible_idx[np.argmin(time_dist)]
    selected_timestamp = timestamp_options[selected_idx]
    return selected_timestamp

################################################################################
#  period_optimizer.py
################################################################################

"""
Implementation of a class for optimizing the cost on a period
"""


CACHE_MAX_SIZE = 2**20

class Period(object):
    """
    Data object for encapsulating all the information that defines a period

    Attributes
    ----------
    * price_sell
    * price_buy
    * load
    * pv
    * balance

    They are all arrays of the same lenght.
    """
    def __init__(self, price_sell, price_buy, load, pv, balance, timesteps=None):
        """All inputs should be arrays"""
        self.price_sell = price_sell
        self.price_buy = price_buy
        self.load = load
        self.pv = pv
        self.balance = balance
        self.timesteps = timesteps
        self._len = len(price_buy)

    def __len__(self):
        return self._len

def get_period_from_df(df):
    """Creates a Period object from a dataframe"""
    price_sell = df.price_sell_00.values
    price_buy = df.price_buy_00.values
    load = df.actual_consumption.values
    pv = df.actual_pv.values
    balance = load - pv
    period = Period(price_sell, price_buy, load, pv, balance)
    return period


class PeriodOptimizer(object):
    """
    Class for optimization of the policy on a period
    """
    def __init__(self, period, battery, allow_end_optimization=True):
        """
        Parameters
        ----------
        df : pandas.Dataframe
            Dataframe with the data of the period that we want to optimize
        battery : Battery
            And object with the properties of the battery
        """
        self.period = period
        self.battery = battery
        self.period_summary = summarize_period(period, battery)
        self._initial_battery_charge = self.battery.current_charge

        self.original_cost = None
        self.baseline_cost = None
        self.policy = None
        self.cost = None
        self.max_charge_variations = self._compute_max_charge_variation()
        self.balance_matches = self._compute_exact_balance_match()
        self.is_low_price = self._compute_is_low_price(3)
        self.is_wait_condition = self._compute_is_low_price(1)
        self.allow_end_optimization = allow_end_optimization

    def optimize(self):
        """
        Returns optimal policy, if the cost is required call get_optimized_cost
        after optimization
        """
        self.cost, self.policy = self._find_best_cost_and_policy(
            self.battery.current_charge, 0, len(self.period_summary))
        return self.policy

    def get_optimized_cost(self):
        if self.original_cost is None:
            self.original_cost = get_original_cost(self.period)
        if self.baseline_cost is None:
            self.baseline_cost = self.get_baseline_cost()
        cost = self.original_cost + self.cost - self.baseline_cost
        return cost

    @lru_cache(maxsize=CACHE_MAX_SIZE)
    def _find_best_cost_and_policy(self, initial_charge, initial_epoch, end_epoch):
        if initial_epoch == end_epoch:
            return 0, []

        target_charge_range = self._get_target_battery_range(initial_charge, initial_epoch)
        costs = []
        paths = []
        for target_charge in target_charge_range:
            cost = self._block_cost(initial_charge, target_charge, initial_epoch)
            ret = self._find_best_cost_and_policy(target_charge, initial_epoch+1, end_epoch)
            costs.append(cost + ret[0])
            paths.append([target_charge] + ret[1])

        min_index = np.argmin(costs)
        return costs[min_index], paths[min_index]

    @lru_cache(maxsize=CACHE_MAX_SIZE)
    def _get_target_battery_range(self, current_charge, block_idx):
        balance_match = self.balance_matches[block_idx]
        target_battery_range = [current_charge + balance_match]

        max_battery_charge, max_battery_discharge = self.max_charge_variations[block_idx]
        if balance_match > 0: # The system is giving energy
            pass
        else: # The system is demanding energy
            # # The following line only has sense if higher prices are expected on following epochs
            if self.is_wait_condition[block_idx]:
                # target_battery_range.append(current_charge)
                if block_idx < len(self.period_summary) - 1:
                    next_balance_match = self.balance_matches[block_idx+1]
                    target_battery = - next_balance_match
                    # Need to check that it is possible to achieve that
                    battery_change = target_battery - current_charge
                    if target_battery - current_charge > 0:
                        battery_change = np.minimum(battery_change, max_battery_charge)
                    else:
                        battery_change = np.maximum(battery_change, max_battery_discharge)
                    target_battery_range.append(current_charge + battery_change)

        if self.is_low_price[block_idx]:
            target_battery_range.append(current_charge + max_battery_charge)

        optimize_end = self.allow_end_optimization and len(self.period_summary) - block_idx < 4
        if optimize_end:
            target_battery_range.append(current_charge + max_battery_discharge)
            target_battery_range.append(current_charge)

        target_battery_range = np.clip(target_battery_range, 0, 1).round(5)
        target_battery_range = np.unique(target_battery_range)
        return target_battery_range

    @lru_cache(maxsize=CACHE_MAX_SIZE)
    def _block_cost(self, current_charge, target_charge, block_idx):
        """
        Computes the cost of the block given the initial and end charge.
        Values are cached for allowing faster dynamic programming
        """
        battery_energy = (target_charge - current_charge)*self.battery.capacity
        if battery_energy > 0:
            battery_energy /= self.battery.charging_efficiency
        else:
            battery_energy *= self.battery.discharging_efficiency
        balance = self.period_summary.balance[block_idx]
        total_balance = balance + battery_energy

        if total_balance < 0:
            price_sell = self.period_summary.price_sell[block_idx]
            cost = total_balance*price_sell
        else:
            price_buy = self.period_summary.price_buy[block_idx]
            cost = total_balance*price_buy
        return cost/1000

    def visualize_optimization(self):
        plot_period_summary_and_policy(self.period_summary.balance, self.period_summary.price_buy,
                                       self.policy, self.battery.capacity,
                                       title='High level policy')

    def visualize_fine_grain_policy(self):
        fine_grain_policy = self.get_fine_grain_policy()
        buy_prices = self.period.price_buy
        sell_prices = self.period.price_sell
        timestep_balances = self.period.balance
        plot_fine_grain_policy(timestep_balances, buy_prices, fine_grain_policy, self.battery.capacity,
                               title='Fine grain policy')

        cost = compute_period_cost(timestep_balances, buy_prices, sell_prices, fine_grain_policy, self.battery)
        print('The cost of the fine grain policy is: %.1f' % cost)

    def get_fine_grain_policy_cost(self):
        fine_grain_policy = self.get_fine_grain_policy()
        buy_prices = self.period.price_buy
        sell_prices = self.period.price_sell
        timestep_balances = self.period.balance
        cost = compute_period_cost(timestep_balances, buy_prices, sell_prices, fine_grain_policy, self.battery)
        return cost

    def get_baseline_cost(self):
        """
        This is the cost that we will get on the clipped energy period if we
        don't use the battery
        """
        cost = 0
        for epoch in range(len(self.period_summary.balance)):
            cost += self._block_cost(0, 0, epoch)
        return cost

    def _compute_max_charge_variation(self):
        """
        max_battery_charge : positive
        max_battery_discharge : negative
        """
        max_charge_variations = []
        for block_idx, _ in enumerate(self.period_summary.balance):
            hours = self.period_summary.timesteps[block_idx]*0.25
            max_battery_charge = hours*self.battery.charging_power_limit/self.battery.capacity*self.battery.charging_efficiency
            max_battery_discharge = hours*self.battery.discharging_power_limit/self.battery.capacity/self.battery.discharging_efficiency
            max_charge_variations.append((max_battery_charge, max_battery_discharge))
        return max_charge_variations

    def _compute_exact_balance_match(self):
        """
        Returns the battery change needed to balance the system demand.
        The balance will be positive when the battery will be charged
        """
        balance_matches = []
        for block_idx, _ in enumerate(self.period_summary.balance):
            battery_change = energy_to_battery_change(-self.period_summary.balance[block_idx], self.battery)
            max_battery_charge, max_battery_discharge = self.max_charge_variations[block_idx]
            if battery_change > 0:
                balance_matches.append(np.minimum(battery_change, max_battery_charge))
            else:
                balance_matches.append(np.maximum(battery_change, max_battery_discharge))
        return balance_matches

    def get_fine_grain_policy(self, n_steps_required=None):
        """
        Computes the amount of charge for each timestep

        Parameters
        ----------
        n_steps : int
            If given it will return the fine_grain_policy once it has equal or more
            steps than required.
        """
        fine_grain_policy = []
        policy = [self._initial_battery_charge] + self.policy
        timestep_idx = 0
        for i, n_steps in enumerate(self.period_summary.timesteps):
            block_timestep_balances = self.period.balance[timestep_idx:timestep_idx + n_steps]
            fine_grain_policy += get_fine_grain_policy_for_block(
                i, policy, self.period_summary.balance, block_timestep_balances, self.battery)
            timestep_idx += n_steps
            if n_steps_required is not None and timestep_idx >= n_steps_required:
                break

        return fine_grain_policy

    def _compute_is_low_price(self, n_blocks):
        prices = self.period_summary.price_buy
        is_low_price = np.zeros_like(prices)
        for i in range(1, n_blocks + 1):
            is_low_price[:-i][prices[:-i]<prices[i:]] = 1
        return is_low_price

    def get_fine_grain_policy_for_timestep(self, timestep_idx, timestep_balance, current_charge):
        policy = [self._initial_battery_charge] + self.policy
        next_charge = get_fine_grain_policy_for_timestep(
            timestep_idx, policy, balances=self.period_summary.balance,
            timestep_balance=timestep_balance, battery=self.battery,
            block_length=self.period_summary.timesteps[0],
            current_charge=current_charge)
        return next_charge


def summarize_period(period, battery=None):
    """
    Returns a summarized version of the period
    The energy balance is clipped to the battery power, however efficiency
    has not been applied

    Parameters
    ----------
    period : Period
        All the information of the period

    Returns
    -------
    balances, prices, timesteps
    """
    # TODO: need to use sell price also for finding division points
    balance, price_buy, price_sell, timesteps = [], [], [], []
    energy_balance = period.balance
    if battery is not None:
        # By multiplying by 0.25 we translate W to Wh
        energy_balance = np.clip(
            energy_balance, battery.discharging_power_limit*0.25, battery.charging_power_limit*0.25)
    start_points, end_points = find_division_points(period)
    for start, end in zip(start_points, end_points):
        balance.append((energy_balance[start:end]).sum())
        price_buy.append(period.price_buy[start])
        price_sell.append(period.price_sell[start])
        timesteps.append(end-start)   #This is because pandas convention

    load, pv = None, None
    period = Period(price_sell, np.asarray(price_buy), load, pv, balance, timesteps)
    return period

def find_division_points(period):
    """
    Finds all the points that allow to divide a period into blocks of
    same prize

    Parameters
    ----------
    period : Period
        All the information of the period
    """
    # TODO: write a test
    price_buy_change_points = _find_price_change_points(period.price_buy)
    price_sell_change_points = _find_price_change_points(period.price_sell)
    balance_change_points = _find_balance_change_points(period)
    division_points = price_buy_change_points + price_sell_change_points + balance_change_points
    division_points = np.arange(len(division_points))[division_points > 0] + 1
    start_points = [0] + division_points.tolist()
    end_points = division_points.tolist() + [len(period)]

    return start_points, end_points

def _find_price_change_points(prices):
    price_change_points = prices[:-1] - prices[1:]
    price_change_points[price_change_points != 0] = 1
    return price_change_points

def _find_balance_change_points(period):
    energy_balance = period.balance
    energy_balance = np.sign(energy_balance)
    balance_change_points = energy_balance[:-1] - energy_balance[1:]
    balance_change_points[balance_change_points != 0] = 1
    return balance_change_points

def _is_a_division_point(period, start):
    """Returns true if the start and end point define a division point"""
    # Prize change
    prices = period.price_buy[start-1:start+1]
    if prices[0] != prices[1]:
        return True
    # Balance change
    energy_balance = period.balance[start-1:start+1]
    energy_balance = np.sign(energy_balance)
    if energy_balance[0] != energy_balance[1]:
        return True
    return False

def compute_period_cost(balances, buy_prices, sell_prices, policy, battery):
    policy = np.asarray([0] + policy)
    battery_change = policy[1:] - policy[:-1]
    battery_energy = [battery_change_to_energy(change, battery) for change in battery_change]
    battery_energy = np.asarray(battery_energy)
    system_balance = balances + battery_energy
    prices = buy_prices.copy()
    prices[system_balance < 0] = sell_prices[system_balance < 0]
    cost = np.sum(prices*system_balance/1000)
    return cost

def get_original_cost(period):
        """
        This is the original cost of the period without battery
        """
        energy_balance = period.balance
        price_buy = period.price_buy
        price_sell = period.price_sell
        price = price_sell.copy()
        price[energy_balance >= 0] = price_buy[energy_balance >= 0]
        cost = np.sum(energy_balance*price)/1000
        return cost

################################################################################
#  fine_grain_policy.py
################################################################################

"""
All the functionality that allows to go from a high level policy to
a fine grain policy
"""


def get_fine_grain_policy_for_block(i, policy, balances, timestep_balances, battery):
    """
    Provides the fine grain policy for a block. A block is a division of the period
    where the system balance and the price is constant.
    """
    battery_change = policy[i+1] - policy[i]
    balance = balances[i]
    initial_charge = policy[i]
    n_steps = len(timestep_balances)

    if battery_change == 0:
        return _keep_initial_charge(n_steps, initial_charge)
    elif balance*battery_change < 0:
        # Then the changes are opposite
        return _sincronize_battery_and_system(
            initial_charge=policy[i], final_charge=policy[i+1], balance=balance,
            battery=battery, timestep_balances=timestep_balances)
    else:
        return _distribute_change_over_timesteps(n_steps, initial_charge=policy[i],
                                                 final_charge=policy[i+1])

def _sincronize_battery_and_system(initial_charge, final_charge, balance, battery, timestep_balances):
    """
    Provides a fine grain policy by sincronizing the battery and the system
    On a first step computes the ratio betwen the energy of the battery and the system.
    Later for each timestep applies the ratio to the energy system, clips to
    battery limit and applies the efficiency.
    """
    battery_energy = battery_change_to_energy(final_charge - initial_charge, battery)
    ratio = battery_energy / balance
    fine_grain_policy = []
    previous_charge = initial_charge
    for timestep_balance in timestep_balances:
        battery_change = _timestep_balance_to_battery_change(timestep_balance, ratio, battery)
        current_charge = previous_charge + battery_change
        fine_grain_policy.append(current_charge)
        previous_charge = current_charge
    fine_grain_policy = np.clip(fine_grain_policy, np.min((initial_charge, final_charge)),
                                np.max((initial_charge, final_charge))).tolist()
    return fine_grain_policy

def _timestep_balance_to_battery_change(timestep_balance, ratio, battery):
    timestep_balance *= ratio
    timestep_balance = apply_battery_power_limit(timestep_balance, 0.25, battery)
    battery_change = energy_to_battery_change(timestep_balance, battery)
    return battery_change

def _keep_initial_charge(n_steps, initial_charge):
    fine_grain_policy = [initial_charge]*n_steps
    return fine_grain_policy

def _distribute_change_over_timesteps(n_steps, initial_charge, final_charge):
    fine_grain_policy = np.linspace(final_charge, initial_charge, n_steps, endpoint=False)
    fine_grain_policy = fine_grain_policy[::-1].tolist()
    assert fine_grain_policy[-1] == final_charge
    return fine_grain_policy

def get_fine_grain_policy_for_timestep(timestep_idx, policy, balances,
                                       timestep_balance, battery, block_length,
                                       current_charge):
    """
    It only works for the initial block

    Returns None when is not able to compute the policy
    """
    block_idx = 0
    initial_charge = policy[block_idx]
    final_charge = policy[block_idx+1]
    battery_change = final_charge - initial_charge
    balance = balances[block_idx]

    remaining_steps = block_length - timestep_idx
    if remaining_steps < 1:
        # Then the block has ended
        return None

    if timestep_balance != 0 and timestep_balance * balance < 0:
        # Then the balance of the system has changed
        return None

    if battery_change == 0:
        return current_charge
    elif balance*battery_change < 0:
        # Then the changes are opposite
        return _sincronize_battery_and_system_for_timestep(
            initial_charge=initial_charge, final_charge=final_charge, balance=balance,
            current_charge=current_charge,
            battery=battery, timestep_balance=timestep_balance)
    else:
        return (final_charge - current_charge)/remaining_steps + current_charge

def _sincronize_battery_and_system_for_timestep(initial_charge, final_charge, current_charge, balance, battery, timestep_balance):
    """
    Provides a fine grain policy by sincronizing the battery and the system
    On a first step computes the ratio betwen the energy of the battery and the system.
    Later for each timestep applies the ratio to the energy system, clips to
    battery limit and applies the efficiency.
    """
    battery_energy = battery_change_to_energy(final_charge - initial_charge, battery)
    ratio = battery_energy / balance
    battery_change = _timestep_balance_to_battery_change(timestep_balance, ratio, battery)
    next_charge = current_charge + battery_change
    next_charge = np.clip(next_charge, np.min((initial_charge, final_charge)),
                                np.max((initial_charge, final_charge))).tolist()
    return next_charge

################################################################################
#  battery_utils.py
################################################################################


def battery_change_to_energy(battery_change, battery):
    """
    Translates a battery change to energy. A positive change is assumed to
    charge the battery
    """
    energy = battery_change*battery.capacity    #Wh
    is_charging = energy > 0
    if is_charging:
        energy /= battery.charging_efficiency
    else:
        energy *= battery.discharging_efficiency
    return energy

def energy_to_battery_change(energy, battery):
    """
    Translates energy to battery change. A positive energy is assumed to
    charge the battery
    """
    is_charging = energy > 0
    if is_charging:
        energy *= battery.charging_efficiency
    else:
        energy /= battery.discharging_efficiency
    battery_change = energy / battery.capacity
    return battery_change

def apply_battery_power_limit(energy, hours, battery):
    return np.clip(energy, battery.discharging_power_limit*hours, battery.charging_power_limit*hours)

################################################################################
#  visualizations_fake.py
################################################################################


def plot_period_summary_and_policy(balances, prices, policy, battery_capacity, title=None):
    pass

def plot_fine_grain_policy(balances, prices, policy, battery_capacity, title=None):
    pass