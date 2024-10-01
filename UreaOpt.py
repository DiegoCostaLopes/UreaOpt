import pyomo.environ as pyo
import pandas as pd
import numpy as np

# data organization

df = pd.read_csv('data/processed/location_db.csv').set_index('id')
df.index = df.index.astype(int)
df = df.loc[(df['location_type'] == 'city') & (df['state'] == 'Sao Paulo')]

locations = list(df.index)
crops = ['cocoa', 'coffee', 'corn', 'rice', 'soy', 'sugarcane']

supply = df[crops].T
demand = df['urea_consumption'].T
routes = ['Pure oxygen gasification',
          'Air mixed gasification',
          'Electrolysis']

data = {'source': crops,
        'Pure oxygen gasification': [0.81 for source in crops],
        'Air mixed gasification': [0.81 for source in crops],
        'Electrolysis': [0.00 for source in crops]}
conversion = pd.DataFrame(data).set_index('source')

cost = pd.DataFrame(index=crops, columns=locations)

for location in locations:
    cost[location] = 80

urea_price = pd.Series(index=locations)
urea_price = pd.Series([310 for location in locations], index=locations)

distance_matrix = pd.read_csv('data/processed/distance_matrix.csv').set_index('origin')
distance_matrix.index = distance_matrix.index.astype(int)
distance_matrix.columns = distance_matrix.columns.astype(int)

distance_matrix = distance_matrix.loc[locations, locations]
print('bla')

#%%

# locations = ['SP',  # tons of bagasse, eucalyptus
#              'MS',  # corn and soy residue, proximity to consumers
#              'NE',  # significant bagasse availability, dedicated wind
#              'RS',  # high availability of rice husks at cheap prices
#              'Intl.',  # fertilizer export or biomass import
#              ]

# biomasses = ['Bagasse',
#              'Sugarcane straw',
#              'Corn stover',

#              'Soybean straw',
#              'Eucalyptus chips',
#              'Eucalyptus bark',
#              'Rice husks', ]

# power_sources = ['Solar',
#                  'Wind',
#                  'Grid', ]
# energy_sources = biomasses + power_sources


# supply = pd.DataFrame({
#     'source': biomasses,
#     'SP': [200000, 100000, 40000, 50000, 100000, 100000, 1000],
#     'MS': [50000, 50000, 200000, 200000, 10000, 10000, 0],
#     'NE': [300000, 300000, 1000, 30000, 40000, 40000, 0],
#     'RS': [10000, 10000, 10000, 1000, 500000, 500000, 500000],
#     'Intl.': [999999, 999999, 999999, 999999, 999999, 999999, 999999],
# }).set_index('source')

# demand = pd.Series({
#     'SP': 200000,
#     'MS': 250000,
#     'NE': 180000,
#     'RS': 150000,
#     'Intl.': 999999,
# })

# routes = ['Pure oxygen gasification',
#           'Air mixed gasification',
#           'Electrolysis']

# data = {'source': energy_sources,
#         'Pure oxygen gasification': [0.81, 0.83, 0.79, 0.75, 0.78, 0.84, 0.80, 0.0, 0.0, 0.0],
#         'Air mixed gasification': [0.81, 0.83, 0.79, 0.75, 0.78, 0.84, 0.80, 0.0, 0.0, 0.0],
#         'Electrolysis': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.2, 1.5, 1.2]}
# conversion = pd.DataFrame(data).set_index('source')  # conversion, in t urea / t biomass (or MWh of electrical power)

# utilities_power = {'source': energy_sources,
#                    'Pure oxygen gasification': []}

# cost = pd.DataFrame({
#     'source': energy_sources,
#     'SP': [50, 50.0, 75, 80, 110, 90, 60, 210, 215, 260],
#     'MS': [65.0, 55, 70, 75, 125, 98, 80, 230, 245, 280],
#     'NE': [80, 80, 90, 85, 125, 95, 75, 195, 200, 230],
#     'RS': [90, 80, 80, 80, 110, 90, 55, 260, 250, 250],
#     'Intl.': [100, 95, 90, 90, 150, 120, 90, 300, 300, 300],
# }).set_index('source')

# urea_price = pd.Series({
#     'SP': 310,
#     'MS': 315,
#     'NE': 320,
#     'RS': 310,  
#     'Intl.': 305,
# })

# distance = pd.DataFrame({
#     'locations': locations,
#     'SP': [100, 1500, 2000, 1000, 200],
#     'MS': [1500, 100, 2500, 2500, 1500],
#     'NE': [2000, 2500, 100, 4000, 200],
#     'RS': [1000, 2500, 4000, 100, 200],
#     'Intl.': [200, 1500, 200, 200, 0],
# }).set_index('locations')


def solve_model(locations, sources, routes, supply, demand, conversion, cost, urea_price, distance):

    capex_ranges = [1, 2, 3, 4]

    # model definition
    m = pyo.ConcreteModel()

    # model sets
    m.LOCATION = pyo.Set(initialize=locations)
    m.SOURCE = pyo.Set(initialize=sources)
    m.ROUTE = pyo.Set(initialize=routes)

    # model parameters
    yearly_hours = 8300
    bigM = 10000000
    transport_price = 6.36 # average price per km
    truck_capacity = 60 # tons per truck, bi-train truck
    price_per_km_ton = transport_price / truck_capacity

    # decision variable definition
    m.biomass_sold = pyo.Var(m.SOURCE, m.LOCATION, m.LOCATION, within=pyo.NonNegativeReals) # biomass B sold from location S to location L
    m.biomass_used = pyo.Var(m.SOURCE, m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)    # biomass B used at route Y at location L
    m.urea_produced = pyo.Var(m.ROUTE, m.LOCATION, within=pyo.NonNegativeReals)             # urea produced via route Y at location L
    m.urea_sold = pyo.Var(m.LOCATION, m.LOCATION, within=pyo.NonNegativeReals)              # urea sold from location L to location LL
    m.plant_installed = pyo.Var(m.LOCATION, within=pyo.Binary)                              # is plant installed at location L?
    m.capex = pyo.Var(within=pyo.NonNegativeReals)                                          # actual capex of the unit
    m.capacity = pyo.Var(within=pyo.NonNegativeReals)                                       # installed capacity of the plant
    m.capex_y = pyo.Var(capex_ranges, within=pyo.Binary)                                    # binary variables for piecewise linearization of CAPEX


    # objective function

    feedstock_cost = sum(m.biomass_used[b, r, l] * cost.loc[b, l] for b in m.SOURCE for r in m.ROUTE for l in m.LOCATION)
    feedstock_transport_cost = sum(m.biomass_sold[b, l1, l2] * price_per_km_ton * distance.loc[l1, l2]
                                for b in m.SOURCE for l1 in m.LOCATION for l2 in m.LOCATION)
    product_transport_cost = sum(m.urea_sold[l1, l2] * price_per_km_ton * distance.loc[l1, l2]
                                for l1 in m.LOCATION for l2 in m.LOCATION)
    urea_revenue = sum(m.urea_sold[l1, l2] * urea_price[l2] for l1 in m.LOCATION for l2 in m.LOCATION)

    m.objective = pyo.Objective(
        expr = urea_revenue - feedstock_cost - feedstock_transport_cost - product_transport_cost,
        sense = pyo.maximize,
    )

    # constraints

    # 1 - total urea produced must be equal to 35 t/h
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
        return sum(m.urea_produced[r, l] for r in m.ROUTE) <= bigM * m.plant_installed[l]

    # 5 - relationship between biomass consumed and urea produced
    @m.Constraint(m.ROUTE, m.LOCATION)
    def biomass_to_urea_ratio(m, r, l):
        # return m.urea_produced[r, l] == sum(m.biomass_used[b, r, l] * conversion.loc[b, r] for b in m.SOURCE)
        return m.urea_produced[r, l] == sum(m.biomass_used[b, r, l] * 0.81 for b in m.SOURCE)

    # 6 - biomass used must be equal to the total biomass sold from all locations
    @m.Constraint(m.SOURCE, m.LOCATION)
    def biomass_sold_to(m, b, l):
        return sum(m.biomass_used[b, r, l] for r in m.ROUTE) == sum(m.biomass_sold[b, l1, l] for l1 in m.LOCATION)

    # total biomass sold from a location must be lower than supply
    @m.Constraint(m.SOURCE, m.LOCATION)
    def biomass_supply_limit(m, b, l):
        return sum(m.biomass_sold[b, l, l2] for l2 in m.LOCATION) <= supply.loc[b, l]

    # total urea sold to a location must be lower

    @m.Constraint(m.LOCATION)
    def urea_demand_limit(m, l):
        return sum(m.urea_sold[l1, l] for l1 in m.LOCATION) <= demand[l]


    # capex and capacity constraints

    #xxx capacity is 35
    @m.Constraint()
    def urea_production_capacity_xxx(m):
        return m.capacity == 70

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

    # @m.Constraint()
    # def urea_production_capacity_51(m):
    #     return 80 - bigM*(1-m.capex_y[80]) <= m.capacity

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


    # @m.Constraint()
    # def urea_capex_41(m):
    #     return (772.35 + (m.capacity - 80)*8.5 - bigM*(1 - m.capex_y[4]) 
    #             <= m.capex)


    # @m.Constraint()
    # def urea_capex_42(m):
    #     return (m.capex <=
    #             772.35 + (m.capacity - 80)*8.5 + bigM*(1 - m.capex_y[4]))




    # model solving
    solver = pyo.SolverFactory('gurobi')
    solver.solve(m)


    print(f'Total revenue = USD{urea_revenue(): ,.0f}')
    print(f'Total feedstock cost = USD{feedstock_cost(): ,.0f}')
    print(f'Total feedstock transport cost = USD{feedstock_transport_cost(): ,.0f}')
    print(f'Total product transport cost = USD{product_transport_cost(): ,.0f}')

    print(f'binary capex array = {[m.capex_y[y]() for y in capex_ranges]}')
    print(f'capex = {m.capex()}')

    return m

m = solve_model(locations, crops, routes, supply, demand, conversion, cost, urea_price, distance_matrix)
    # results conversion into useful formats
    # noinspection PyCallingNonCallable
plant_installed = pd.Series(
    {i: j for i, j in zip(m.LOCATION, [m.plant_installed[l]() for l in m.LOCATION])}
)

L = plant_installed.index[plant_installed >= 0.999].to_list()

biomass_used = pd.DataFrame(
    {r:      {b: k for b, k in zip(m.SOURCE, [m.biomass_used[b, r, L]() for b in m.SOURCE])} for r in m.ROUTE}
)
biomass_sold = pd.DataFrame(
    {l1:     {b: k for b, k in zip(m.SOURCE, [m.biomass_sold[b, l1, L]() for b in m.SOURCE])} for l1 in m.LOCATION}
)

urea_sold = pd.Series(
    {i: j for i, j in zip(m.LOCATION, [m.urea_sold[L, l2]() for l2 in m.LOCATION])}
)
