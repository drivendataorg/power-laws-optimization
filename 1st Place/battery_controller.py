
from ortools.linear_solver import pywraplp
 
class BatteryContoller(object):
    step = 960
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


        self.step -= 1
        if (self.step == 1): return 0
        if (self.step > 1): number_step = min(96, self.step)
        #
        price_buy = price_buy.tolist()
        price_sell = price_sell.tolist()
        load_forecast = load_forecast.tolist()
        pv_forecast = pv_forecast.tolist()  
        #
        energy = [None] * number_step

        for i in range(number_step):
            if (pv_forecast[i] >=50): energy[i] = load_forecast[i] - pv_forecast[i]
            else: energy[i] = load_forecast[i]
        #battery
        capacity = battery.capacity
        charging_efficiency = battery.charging_efficiency
        discharging_efficiency = 1. / battery.discharging_efficiency
        current = capacity * battery.current_charge 
        limit = battery.charging_power_limit
        dis_limit = battery.discharging_power_limit
        limit /= 4.
        dis_limit /= 4.

        # Ortools
        solver = pywraplp.Solver("B", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
         
        #Variables: all are continous
        charge = [solver.NumVar(0.0, limit, "c"+str(i)) for i in range(number_step)] 
        dis_charge = [solver.NumVar( dis_limit, 0.0, "d"+str(i)) for i in range(number_step)]
        battery_power = [solver.NumVar(0.0, capacity, "b"+str(i)) for i in range(number_step+1)]
        grid = [solver.NumVar(0.0, solver.infinity(), "g"+str(i)) for i in range(number_step)] 
         
        #Objective function
        objective = solver.Objective()
        for i in range(number_step):
            objective.SetCoefficient(grid[i], price_buy[i] - price_sell[i])
            objective.SetCoefficient(charge[i], price_sell[i] + price_buy[i] / 1000.)
            objective.SetCoefficient(dis_charge[i], price_sell[i])             
        objective.SetMinimization()
         
        # 3 Constraints
        c_grid = [None] * number_step
        c_power = [None] * (number_step+1)
         
        # first constraint
        c_power[0] = solver.Constraint(current, current)
        c_power[0].SetCoefficient(battery_power[0], 1)
         
        for i in range(0, number_step):
            # second constraint
            c_grid[i] = solver.Constraint(energy[i], solver.infinity())
            c_grid[i].SetCoefficient(grid[i], 1)
            c_grid[i].SetCoefficient(charge[i], -1)
            c_grid[i].SetCoefficient(dis_charge[i], -1)
            # third constraint
            c_power[i+1] = solver.Constraint( 0, 0)
            c_power[i+1].SetCoefficient(charge[i], charging_efficiency)
            c_power[i+1].SetCoefficient(dis_charge[i], discharging_efficiency)
            c_power[i+1].SetCoefficient(battery_power[i], 1)
            c_power[i+1].SetCoefficient(battery_power[i+1], -1)

        #solve the model
        solver.Solve()

        if ((energy[0] < 0) & (dis_charge[0].solution_value() >= 0)):
            n = 0
            first = -limit
            mid = 0

            sum_charge = charge[0].solution_value()
            last = energy[0]
            for n in range(1, number_step):
                if((energy[n] > 0) | (dis_charge[n].solution_value() < 0) | (price_sell[n] != price_sell[n-1])):
                    break
                last = min(last, energy[n])
                sum_charge += charge[n].solution_value()
            if (sum_charge <= 0.):
                 return battery_power[1].solution_value() / capacity
            def tinh(X):
                res = 0
                for i in range(n):
                    res += min(limit, max(-X - energy[i], 0.))
                if (res >= sum_charge): return True
                return False 
            last = 2 - last
            # binary search
            while (last - first > 1):
                mid = (first + last) / 2
                if (tinh(mid)): first = mid
                else: last = mid
            return (current + min(limit, max(-first - energy[0] , 0)) * charging_efficiency) / capacity
        
        if ((energy[0] > 0) & (charge[0].solution_value() <=0)):
            n = 0
            first = dis_limit
            mid = 0
            sum_discharge = dis_charge[0].solution_value()
            last = energy[0]
            for n in range(1, number_step):
                if ((energy[n] < 0) | (charge[n].solution_value() > 0) | (price_sell[n] != price_sell[n-1]) | (price_buy[n] != price_buy[n-1])):
                    break
                last = max(last, energy[n])
                sum_discharge += dis_charge[n].solution_value()
            if (sum_discharge >= 0.): 
                return battery_power[1].solution_value() / capacity

            def tinh2(X):
                res = 0
                for i in range(n):
                    res += max(dis_limit, min(X - energy[i], 0))
                if (res <= sum_discharge): return True
                return False                      
            last += 2

            # binary search
            while (last - first > 1):
                mid = (first + last) / 2
                if (tinh2(mid)): first = mid
                else: last = mid
            return (current +  max(dis_limit, min(first - energy[0], 0)) * discharging_efficiency) / capacity
        return battery_power[1].solution_value() / capacity