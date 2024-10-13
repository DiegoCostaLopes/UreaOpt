
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import time

start_time = time.time()

# data organization

biomass = ['rice_husk', 'coffee_husk', 'corn_stover', 'soy_straw', 'sugarcane_straw', 'sugarcane_bagasse']
power = ['power_grid']
sources = biomass + power
sources_price = [source + '_price' for source in sources] 
# full db has more data on area planted and etc. that the algorithm and
# the results dont necessarily need.
columns = (['name', 'location_type', 'region', 'state', 'location_id', 'urea_demand', 'urea_price']
            + sources + sources_price)

data = pd.read_pickle('data/processed/location_db.p').loc[:, columns]
distance_matrix = pd.read_pickle('data/processed/distance_matrix.p')

# choosing which subset of data will I study.
# data = data.loc[(data['location_type'] == 'city') & (data['state'] == 'Sao Paulo')]
# data = data.loc[(data['location_type'] == 'microregion')]
data = data.loc[(data['location_type'] == 'microregion') & (data['region'] == 'Centro-Oeste')]
locations = list(data.index)
correspondence_series = pd.Series(data.index, index=data['location_id'])
distance_matrix = distance_matrix.rename(index=correspondence_series, columns=correspondence_series)
distance_matrix = distance_matrix.loc[locations, locations] + 100 # correcting for transportation inside the same city

routes = ['Pure oxygen gasification',
          'Air mixed gasification', 
          'Electrolysis']

# biomass or power consumption that goes directly to urea in the form of feedstock
conversion_data = {'source': biomass + ['power'],
        'Pure oxygen gasification': [0.7616, 0.7616, 0.7616, 0.7616, 0.7616, 0.571209] + [0.0],
        'Air mixed gasification': [0.727710224774961, 0.727710224774961, 0.727710224774961, 0.727710224774961, 0.727710224774961, 0.545782163392111] + [0.0],
        'Electrolysis': [0.00 for b in biomass] + [0.00]}
conversion = pd.DataFrame(conversion_data).set_index('source')

# biomass or power that goes to utility (in the form of steam or power)
utility_data = {'source': biomass + ['power'],
        'Pure oxygen gasification': [0.294430367536132, 0.294430367536132, 0.294430367536132, 0.294430367536132, 0.294430367536132, 0.326134953063215] + [0.0],
        'Air mixed gasification': [0.250124450940096, 0.250124450940096, 0.250124450940096, 0.250124450940096, 0.250124450940096, 0.277058126680156] + [0.0],
        'Electrolysis': [0.00 for b in biomass] + [0.00]}
utility = pd.DataFrame(utility_data).set_index('source')

def solve_model(data, biomass, power, routes, conversion, utility, distance):

    capex_ranges = [1, 2, 3, 4]
    sources = biomass + ['power']

    locations = list(data.index)
    supply = data[biomass+power]
    demand = data['urea_demand']
    urea_price = data['urea_price']
    prices = data[[source+'_price' for source in biomass+power]]
    prices.columns = biomass+power

    # model definition
    m = pyo.ConcreteModel()

    # model sets
    m.LOCATION = pyo.Set(initialize=locations)
    m.BIOMASS = pyo.Set(initialize=biomass)
    m.POWER = pyo.Set(initialize=power)
    m.SOURCE = pyo.Set(initialize=sources)
    m.ROUTE = pyo.Set(initialize=routes)

    # model parameters
    yearly_hours = 8300
    bigM = 10000
    transport_price = 6.36 / 5.5 * 2# average price per km
    truck_capacity = 40 # tons per truck, bi-train truck
    price_per_km_ton = transport_price / truck_capacity

    # decision variable definition
    
    m.biomass_sold = pyo.Var(m.BIOMASS, m.LOCATION, m.LOCATION, within=pyo.NonNegativeReals)# biomass B sold from location S to location L
    m.power_sold = pyo.Var(m.POWER, m.LOCATION, within=pyo.NonNegativeReals)                # power sold at location L

    m.energy_source_used = pyo.Var(m.SOURCE, m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)
    m.biomass_used = pyo.Var(m.BIOMASS, m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)
    m.power_used = pyo.Var(m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)
    m.utility_used = pyo.Var(m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)

    m.urea_produced = pyo.Var(m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)             # urea produced via route Y at location L
    m.urea_sold = pyo.Var(m.LOCATION, m.LOCATION, within=pyo.NonNegativeReals)              # urea sold from location L to location LL
    m.plant_installed = pyo.Var(m.LOCATION, within=pyo.Binary)                              # is plant installed at location L?
    m.capex = pyo.Var(within=pyo.NonNegativeReals)                                          # actual capex of the unit
    m.capacity = pyo.Var(within=pyo.NonNegativeReals)                                       # installed capacity of the plant
    m.capex_y = pyo.Var(capex_ranges, within=pyo.Binary)                                    # binary variables for piecewise linearization of CAPEX

    # objective function

    
    m.biomass_cost = (sum(m.biomass_sold[b, l1, l2] * prices.loc[l1 , b] for b in m.BIOMASS for l1 in m.LOCATION for l2 in m.LOCATION))
    m.power_cost = sum(m.power_sold[p, l] * prices.loc[l, p] for p in m.POWER for l in m.LOCATION)
    m.biomass_transport_cost = sum(m.biomass_sold[b, l1, l2] * price_per_km_ton * distance.loc[l1, l2]
                                for b in m.BIOMASS for l1 in m.LOCATION for l2 in m.LOCATION)
    m.product_transport_cost = sum(m.urea_sold[l1, l2] * price_per_km_ton * distance.loc[l1, l2]
                                for l1 in m.LOCATION for l2 in m.LOCATION)
    m.urea_revenue = sum(m.urea_sold[l1, l2] * urea_price[l2] for l1 in m.LOCATION for l2 in m.LOCATION)

    m.cash_flow = m.urea_revenue - m.biomass_cost - m.power_cost - m.biomass_transport_cost - m.product_transport_cost
    periods = list(range(1,21)) # 20 years
    discount_rate = 0.10
    m.NPV = sum(m.cash_flow / (1 + discount_rate)**t for t in periods) - m.capex*1000000

    m.objective = pyo.Objective(
        expr = m.NPV,
        sense = pyo.maximize,
    )

    # constraints

    # 1 - total urea produced must be equal to its capacity
    @m.Constraint()
    def total_urea_produced(m):
        return sum(m.urea_produced[r, l] for r in m.ROUTE for l in m.LOCATION) == m.capacity * yearly_hours

    # 2 - total urea produced at one site must be equal to total sold
    @m.Constraint(m.LOCATION)
    def total_urea_sold(m, l1):
        return (sum(m.urea_produced[r, l1] for r in m.ROUTE) ==
                sum(m.urea_sold[l1, l2] for l2 in m.LOCATION))

    # 3 - plant can only be installed in one location

    @m.Constraint()
    def plant_location_limit(m):
        return sum(m.plant_installed[l] for l in m.LOCATION) == 1

    # 4 - if plant is not installed in the location, then production must be zero
    @m.Constraint(m.LOCATION)
    def plant_production_limit(m, l):
        return sum(m.urea_produced[r, l] for r in m.ROUTE) <= bigM * 8300 * m.plant_installed[l]

    # 5 - relationship between energy source consumed and urea produced
    @m.Constraint(m.ROUTE, m.LOCATION)
    def biomass_to_urea_ratio(m, r, l):
        # return m.urea_produced[r, l] == sum(m.biomass_used[b, r, l] * conversion.loc[b, r] for b in m.SOURCE)
        return m.urea_produced[r, l] == sum(m.energy_source_used[s, r, l] * conversion.loc[s, r] for s in m.SOURCE)
  
    # 6 - relationship between utility used and energy source consumed
    @m.Constraint(m.ROUTE, m.LOCATION)
    def utility_to_urea_ratio(m, r, l):
        return m.utility_used[r, l] == sum(m.energy_source_used[s, r, l]  * utility.loc[s, r] for s in m.SOURCE)

    # 7 - biomass used must be equal to the total biomass sold from all locations
    @m.Constraint(m.BIOMASS, m.LOCATION)
    def biomass_sold_to(m, b, l):
        return sum(m.energy_source_used[b, r, l] for r in m.ROUTE) == sum(m.biomass_sold[b, l1, l] for l1 in m.LOCATION)
    
    # 8 - power used must be equal to the total biomass sold from all locations
    @m.Constraint(m.LOCATION)
    def power_sold_to(m, l):
        return sum(m.energy_source_used['power', r, l] + m.utility_used[r, l]  for r in m.ROUTE) == sum(m.power_sold[p, l] for p in m.POWER)

    # 9 - total biomass sold from a location must be lower than supply
    @m.Constraint(m.BIOMASS, m.LOCATION)
    def biomass_supply_limit(m, b, l):
        return sum(m.biomass_sold[b, l, l2] for l2 in m.LOCATION) <= supply.loc[l, b]

    # total urea sold to a location must be lower

    @m.Constraint(m.LOCATION)
    def urea_demand_limit(m, l):
        return sum(m.urea_sold[l1, l] for l1 in m.LOCATION) <= demand[l]

    # capex and capacity constraints

    # @m.Constraint()
    # def urea_production_capacity_xxx(m):
    #     return m.capacity == 35

    # 1 - if capacity is between 0 and x t/h, then its binary variable is 1
    @m.Constraint()
    def urea_production_capacity_11(m):
        return 0 <= m.capacity
 
    @m.Constraint()
    def urea_production_capacity_12(m):
        return m.capacity <= 20 + bigM*(1 - m.capex_y[1])

    @m.Constraint()
    def urea_production_capacity_21(m):
        return 20 - bigM*(1 - m.capex_y[2]) <= m.capacity

    @m.Constraint()
    def urea_production_capacity_22(m):
        return m.capacity <= 35 + bigM*(1 - m.capex_y[2])

    @m.Constraint()
    def urea_production_capacity_31(m):
        return 35 - bigM*(1 - m.capex_y[3]) <= m.capacity

    @m.Constraint()
    def urea_production_capacity_32(m):
        return m.capacity <= 50 + bigM*(1 - m.capex_y[3])


    @m.Constraint()
    def urea_production_capacity_41(m):
        return 50 - bigM*(1 - m.capex_y[4]) <= m.capacity

    @m.Constraint()
    def urea_production_capacity_42(m):
        return m.capacity <= 80 + bigM*(1 - m.capex_y[4])

    @m.Constraint()
    def urea_production_capacity_7(m):
        return sum(m.capex_y[i] for i in capex_ranges) == 1

    # # 2 - capex is equal to its category if its binary variable is 1

    @m.Constraint()
    def urea_capex_11(m):
        return 11.9*m.capacity - bigM*(1 - m.capex_y[1]) <= m.capex

    @m.Constraint()
    def urea_capex_12(m):
        return m.capex <= 11.9*m.capacity + bigM*(1 - m.capex_y[1])

    @m.Constraint()
    def urea_capex_21(m):
        return 231.91 + (m.capacity - 20)*9.12 - bigM*(1 - m.capex_y[2]) <= m.capex

    @m.Constraint()
    def urea_capex_22(m):
        return m.capex <= 231.91 + (m.capacity - 20)*9.12 + bigM*(1 - m.capex_y[2])

    @m.Constraint()
    def urea_capex_31(m):
        return (368.7 + (m.capacity - 35)*8.946 - bigM*(1 - m.capex_y[3]) 
                <= m.capex)

    @m.Constraint()
    def urea_capex_32(m):
        return (m.capex <=
                368.7 + (m.capacity - 35)*8.946 + bigM*(1 - m.capex_y[3]))

    @m.Constraint()
    def urea_capex_41(m):
        return (502.89 + (m.capacity - 50)*8.98 - bigM*(1 - m.capex_y[4]) 
                <= m.capex)

    @m.Constraint()
    def urea_capex_42(m):
        return (m.capex <=
                502.89 + (m.capacity - 50)*8.98 + bigM*(1 - m.capex_y[4]))

    print('Model building done! Initializing solver...')

    # model solving
    solver = pyo.SolverFactory('gurobi')
    solver.solve(m)
    print(f'Model solving complete!')
    return m

m = solve_model(data, biomass, power, routes, conversion, utility, distance_matrix)

# model solving complete. converting results into useful formats...
print(f'Plant capacity: {m.capacity()} t/h')
print(f'Total revenue = USD{m.urea_revenue(): ,.0f}')
print(f'Total biomass cost = USD{m.biomass_cost(): ,.0f}')
print(f'Total power cost = USD{m.power_cost(): ,.0f}')
print(f'Total biomass transport cost = USD{m.biomass_transport_cost(): ,.0f}')
print(f'Total product transport cost = USD{m.product_transport_cost(): ,.0f}')
print(f'Net cash flow = USD{m.cash_flow(): ,.0f}')
print(f'NPV = USD {m.NPV(): ,.0f}')    
print(f'Plant CAPEX = {m.capex()}')

plant_installed = pd.Series([m.plant_installed[l1]() for l1 in m.LOCATION],
                             name='plant_installed', index=locations)
L = plant_installed.idxmax()
energy_source_used = pd.DataFrame([[m.energy_source_used[s, r, L]() for s in m.SOURCE] for r in m.ROUTE],
                            index=routes, columns=sources)
biomass_sold = pd.DataFrame([[m.biomass_sold[b, l1, L]() for b in m.BIOMASS] for l1 in m.LOCATION],
                            index=locations, columns=[b+'_used' for b in biomass])
power_sold = pd.DataFrame([[m.power_sold[p, l1]() for p in m.POWER] for l1 in m.LOCATION],
                            index=locations, columns=[p+'_used' for p in power])

utility_used = pd.DataFrame([[m.utility_used[r, l]() for r in m.ROUTE] for l in m.LOCATION], index=locations, columns=routes)

urea_produced = pd.DataFrame([[m.urea_produced[r, l]() for r in m.ROUTE] for l in m.LOCATION], index=locations, columns=routes)
urea_sold = pd.Series([m.urea_sold[L, l2]() for l2 in m.LOCATION], name='urea_sold', index=locations)
data = pd.concat([data, plant_installed, biomass_sold, urea_sold], axis=1)

file_name = 'microregion_CO'
export_model = True
if export_model:
    data.to_pickle('data/results/main_results_' + file_name + '.p')
    energy_source_used.to_pickle('data/results/energy_source_used_'+ file_name +'.p')

end_time = time.time()
duration = (end_time - start_time) / 60
print(f'Run complete! Time to solve: {duration} mins')
