
# ==============================================================================
#    THIS CLASS WILL BE IMPLEMENTED BY COMPETITORS
# ==============================================================================
class BatteryContoller(object):
    """ The BatteryContoller class handles providing a new "target state of charge"
        at each time step.

        This class is instantiated by the simulation script, and it can
        be used to store any state that is needed for the call to
        propose_state_of_charge that happens in the simulation.

        The propose_state_of_charge method returns the state of
        charge between 0.0 and 1.0 to be attained at the end of the coming
        quarter, i.e., at time t+15 minutes.
        The arguments to propose_state_of_charge are as follows:
        - The current time
        - The battery (see battery.py for useful properties, including current_charge and capacoty)
        - The actual load of the previous quarter.
        - The actual PV production of the previous quarter.
        - The price at which electricity can be bought from the grid for the
          next 96 quarters (i.e., an array of 96 values).
        - The price at which electricity can be sold to the grid for the
          next 96 quarters (i.e., an array of 96 values). This is often 0.
        - The forecast of the load (consumption) established at time t for the next 96
          quarters (i.e., an array of 96 values).
        - The forecast of the PV production established at time t for the next
          96 quarters (i.e., an array of 96 values).
    """
    def propose_state_of_charge(self,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):

        # return the proposed state of charge ...
        return 1.0
