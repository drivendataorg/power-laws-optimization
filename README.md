<a href="https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/">
    <img src="https://s3.amazonaws.com/drivendata-public-assets/se-challenge-3-banner.jpg"/>
</a>

# Power Laws: Optimizing Demand-side Strategies

Flexibility can be defined as "the ability of a resource, whether any component or collection of components of the power system, to respond to the known and unknown changes of power system conditions at various operational timescales".1 The exploitation of flexibility is essential to avoid costly reinforcements of the power system and maintain security of supply while increasing the penetration of renewable (and intermittent) sources of energy.

Flexibility can be produced in different manners. It might come from generation options, from energy storage or from energy demand. In some cases, generation can also be proposed through alternative dispatchable assets such as Combined Heat and Power (CHP). Storage is valid for both electricity and heat. Energy storage is an easy way to increase building flexibility, provided there is a business case for such an investment. The present challenge is focused on making a good usage of an installed storage system.

Viewed from the demand side, in the case of smart buildings, time of use tariffs incite to use energy when it is the most available. Given such a tariff, the goal is to buy more energy when its price is the lowest, and buy less (or possibly sell) energy when its price is the highest.

The goal in this competition is to build an algorithm that controls a battery charging system and spends the least amount of money over a simulation period.

## What's in this Repository

This repository contains code from winning competitors in the [Power Laws: Optimizing Demand-side Strategies](https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

This repository also contains the simulation engine code provided to competitors for testing their solutions. You can find the simulation engine code in the [`simulation_engine/`](simulation_engine/) subdirectory—see that subdirectory's README for more information.

## Final Results

Place |Team or User | Score | Summary of Model
--- | --- | --- | ---
1 | VietNam national ORlab | -0.201322 | We considered the problem as a dynamic optimization problem. The problem at each step was modeled as a linear programming (LP). We selected Ortools to solve LP model optimally because it seemed to be the fastest and easy to install on docker.
2 | ironbar | -0.199243 | My solution is based on simplificatoin of the period, dynamic programming, and intelligent pruning of the actions.
3 | Helios | -0.198155 | Formulate a linear programming model for the optimization problem at each step with forecast data and use an open-source tool, which can be installed by pip to solve it. Scatter the energy charged (or discharged) among steps to avoid buying superfluous energy due to the uncertainty of next forecasts.

**[Interview with winners](https://drivendata.co/blog/power-laws-optimization-winners/)**

---

<a href="http://www.drivendata.org">
    <img src="https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png"/>
</a>

