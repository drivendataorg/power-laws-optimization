
from ortools.linear_solver import pywraplp

class BatteryContoller(object):
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
        self.id = self.id + 1
        n = max( 1, min( 96, 960 - self.id))
        solver = pywraplp.Solver('Battery_controller', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        infinity = solver.infinity()
        # Variables
        capa = battery.capacity
        c_p_l =  ( battery.charging_power_limit * battery.charging_efficiency) / 4.0
        d_p_l = ( battery.discharging_power_limit / battery.discharging_efficiency) / 4.0
        
        X = [ solver.NumVar(0.0, capa, "X" + str( i)) for i in range( n+1)]
        Y = [ solver.NumVar(0.0, c_p_l, "Y"+ str( i)) for i in range( n)]
        Z = [ solver.NumVar(0.0, infinity, "Z"+ str( i)) for i in range( n)]


        eff = 1.0 / battery.charging_efficiency
        dis_eff =  battery.discharging_efficiency
        price_buy = price_buy.tolist()
        price_sell = price_sell.tolist()
        load_forecast = load_forecast.tolist()
        pv_forecast = pv_forecast.tolist()
        
        objective = solver.Objective()

        objective.SetCoefficient( X[n], price_sell[n-1] * dis_eff)

        for i in range( n): 
            objective.SetCoefficient( Z[i],  price_buy[i] - price_sell[i])
            objective.SetCoefficient( Y[i], price_buy[i]/2000.0 + price_sell[i] * ( eff - dis_eff))
            if( i>=1): objective.SetCoefficient( X[i], (price_sell[i-1] - price_sell[i]) * dis_eff)
        objective.SetMinimization()
        # Constraints
        g_constraint = [None]*n
        ec_constraint = [solver.Constraint( 0.0, infinity) for i in range( n)]
        x_constraint = [solver.Constraint( d_p_l , infinity) for i in range( n)]

        init = solver.Constraint( battery.current_charge * capa, battery.current_charge * capa)
        init.SetCoefficient( X[0], 1.0)
        for i in range( n):
            x_constraint[i].SetCoefficient( X[i+1], 1.0)
            x_constraint[i].SetCoefficient( X[i], -1.0)

            ec_constraint[i].SetCoefficient( Y[i], 1.0)
            ec_constraint[i].SetCoefficient( X[i+1], -1.0)
            ec_constraint[i].SetCoefficient( X[i], 1.0)

            if( pv_forecast[i]<20): pv_forecast[i] = 0.0
            
            g_constraint[i] = solver.Constraint( load_forecast[i] - pv_forecast[i], infinity)        
            g_constraint[i].SetCoefficient( Z[i], 1.0)
            g_constraint[i].SetCoefficient( Y[i], dis_eff - eff)
            g_constraint[i].SetCoefficient( X[i+1], -dis_eff)
            g_constraint[i].SetCoefficient( X[i], dis_eff)
        # Solver

        solver.Solve()
        
        if(  ( X[1].solution_value() >= X[0].solution_value()) & (pv_forecast[0] > load_forecast[0])):
            charging = X[1].solution_value() - X[0].solution_value()
            energy = pv_forecast[0] - load_forecast[0]
            for i in range( 1, n): 
                if( ( price_sell[i] != price_sell[i-1]) | ( pv_forecast[i] < load_forecast[i]) | ( X[i].solution_value() > X[i+1].solution_value()) ): break
                energy = energy + pv_forecast[i] - load_forecast[i]
                charging = charging + X[i+1].solution_value() - X[i].solution_value()
            if( energy <= 0.0): return X[1].solution_value() / capa
            return battery.current_charge + ( ( pv_forecast[0] - load_forecast[0]) * (charging / energy)) / capa


        if(  ( X[1].solution_value() <= X[0].solution_value()) & (pv_forecast[0] < load_forecast[0])):
            charging = X[0].solution_value() - X[1].solution_value()
            energy = load_forecast[0] - pv_forecast[0]
            for i in range( 1, n): 
                if( ( price_sell[i] != price_sell[i-1]) | ( price_buy[i] != price_buy[i-1]) | ( pv_forecast[i] > load_forecast[i]) | ( X[i].solution_value() < X[i+1].solution_value()) ): break
                energy = energy + load_forecast[i] - pv_forecast[i]
                charging = charging + X[i].solution_value() - X[i+1].solution_value()
            if( energy <= 0.0): return X[1].solution_value() / capa
            return battery.current_charge + ( ( pv_forecast[0] - load_forecast[0]) * (charging / energy)) / capa
        return  X[1].solution_value() / capa
    id = 0