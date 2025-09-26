# main_lca_calculator.py

class MetalLCA:
    """
    A class to calculate a simplified Life Cycle Assessment (LCA) for a metal product.
    This model focuses on Global Warming Potential (GWP) as the primary impact category.

    Disclaimer: The emission factors used here are for demonstration purposes only and
    do not represent scientifically validated data for any specific process. A real
    LCA requires a comprehensive database like ecoinvent.
    """

    EMISSION_FACTORS = {
        # Impact per unit of activity (kg CO2eq)
        'energy': {
            'grid_electricity_kwh': 0.45,  # kg CO2eq per kWh
            'mj_from_coal': 0.09,          # kg CO2eq per MJ
            'mj_from_gas': 0.05,           # kg CO2eq per MJ
            'mj_from_renewables': 0.005,   # kg CO2eq per MJ (includes manufacturing of panels/turbines)
            'mj_from_hydro': 0.004,        # kg CO2eq per MJ
        },
        'transport': {
            'truck': 0.1,  # kg CO2eq per ton-km
            'train': 0.03, # kg CO2eq per ton-km
            'ship': 0.01,  # kg CO2eq per ton-km
        },
        'materials': {
            'explosives_kg': 2.5, # kg CO2eq per kg of explosive
        },
        'waste': {
            'landfill_ton': 25, # kg CO2eq per ton of landfilled waste (methane emissions, etc.)
        },
        'direct_emissions': {
            'co_kg': 1.57,    # GWP of CO over 100 years
            'so2_kg': -0.38,  # Negative GWP (cooling effect of sulfate aerosols)
            'nox_kg': 7.0,    # N2O is a potent GHG, NOx is a precursor
        }
    }

    def __init__(self, **kwargs):
        """Initializes the LCA calculator with all necessary parameters."""
        self.params = kwargs
        self.results = {}

    def _get_param(self, key, default=0):
        """Helper to safely get a parameter value."""
        return self.params.get(key, default)

    def _calculate_transport_gwp(self, distance_km, weight_tons, mode):
        """Calculates GWP from transportation."""
        mode = str(mode).lower()
        if mode not in self.EMISSION_FACTORS['transport']:
            print(f"Warning: Transport mode '{mode}' not found. Defaulting to 'truck'.")
            mode = 'truck'
        factor = self.EMISSION_FACTORS['transport'][mode]
        return distance_km * weight_tons * factor

    def calculate_extraction_gwp(self):
        """Calculates GWP from raw material extraction and transport."""
        # Energy from mining
        energy_gwp = self._get_param('Energy_Mining_MJ') * self.EMISSION_FACTORS['energy']['mj_from_gas'] # Assuming gas power

        # Explosives and chemicals
        explosives_gwp = self._get_param('Explosives_Chemicals_kg') * self.EMISSION_FACTORS['materials']['explosives_kg']

        # Transport of raw ore
        transport_gwp = self._calculate_transport_gwp(
            self._get_param('Transport_Raw_km'),
            self._get_param('Quantity_tons'),
            self._get_param('Transport_Raw_Mode')
        )

        total_extraction_gwp = energy_gwp + explosives_gwp + transport_gwp
        return total_extraction_gwp

    def calculate_manufacturing_gwp(self):
        """Calculates GWP from the manufacturing and processing stage."""
        total_energy_mj = self._get_param('Energy_Consumption_MJ')
        
        # Weighted average emission factor for fuel mix
        fuel_coal_pct = self._get_param('Fuel_Coal%') / 100.0
        fuel_gas_pct = self._get_param('Fuel_Gas%') / 100.0
        fuel_renewables_pct = self._get_param('Fuel_Renewables%') / 100.0
        fuel_hydro_pct = self._get_param('Fuel_Hydro%') / 100.0

        weighted_fuel_factor = (
            fuel_coal_pct * self.EMISSION_FACTORS['energy']['mj_from_coal'] +
            fuel_gas_pct * self.EMISSION_FACTORS['energy']['mj_from_gas'] +
            fuel_renewables_pct * self.EMISSION_FACTORS['energy']['mj_from_renewables'] +
            fuel_hydro_pct * self.EMISSION_FACTORS['energy']['mj_from_hydro']
        )
        
        fuel_gwp = total_energy_mj * weighted_fuel_factor
        
        # Electricity GWP
        electricity_gwp = self._get_param('Electricity_kWh') * self.EMISSION_FACTORS['energy']['grid_electricity_kwh']
        
        # Direct emissions GWP
        direct_emissions_gwp = (
            self._get_param('Emissions_CO_kg') * self.EMISSION_FACTORS['direct_emissions']['co_kg'] +
            self._get_param('Emissions_SO2_kg') * self.EMISSION_FACTORS['direct_emissions']['so2_kg'] +
            self._get_param('Emissions_NOx_kg') * self.EMISSION_FACTORS['direct_emissions']['nox_kg']
        )
        
        return fuel_gwp + electricity_gwp + direct_emissions_gwp

    def calculate_use_phase_gwp(self):
        """Calculates GWP from the product's use phase."""
        # Energy consumed during the product's entire lifetime
        total_use_energy_mj = self._get_param('Energy_Use_MJ') * self._get_param('Product_Lifetime_years')
        # Assuming energy during use comes from the grid
        use_gwp = total_use_energy_mj * (self.EMISSION_FACTORS['energy']['grid_electricity_kwh'] / 3.6) # Convert kWh factor to MJ
        return use_gwp

    def calculate_eol_gwp(self):
        """Calculates GWP from End-of-Life, including recycling credit."""
        total_mass = self._get_param('Quantity_tons')
        
        # Transport to disposal/recycling facility
        transport_gwp = self._calculate_transport_gwp(
            self._get_param('Transport_EOL_km'),
            total_mass,
            'truck' # Assuming truck for EOL transport
        )
        
        # GWP from landfilling
        landfilling_mass = total_mass * (self._get_param('Landfilling_Rate%') / 100.0)
        landfill_gwp = landfilling_mass * self.EMISSION_FACTORS['waste']['landfill_ton']
        
        # Recycling Credit (Avoided Burden)
        # The benefit of recycling is avoiding the extraction of new materials.
        # This is represented as a negative GWP value.
        mass_recycled = total_mass * (self._get_param('Recycling_Rate%') / 100.0)
        
        # We assume the impact of extracting 1 ton of virgin material is the GWP we calculated in that stage
        # divided by the quantity produced. This is a simplification.
        gwp_per_ton_virgin = self.results.get('extraction', 0) / self._get_param('Quantity_tons', 1)
        
        recycling_credit = -1 * mass_recycled * gwp_per_ton_virgin
        
        return transport_gwp + landfill_gwp + recycling_credit

    def run_lca(self):
        """Runs the full LCA calculation and stores the results."""
        self.results['extraction'] = self.calculate_extraction_gwp()
        self.results['manufacturing'] = self.calculate_manufacturing_gwp()
        self.results['use_phase'] = self.calculate_use_phase_gwp()
        # EOL must be calculated last to use the extraction results for the credit
        self.results['end_of_life'] = self.calculate_eol_gwp()
        
        self.results['total_gwp'] = sum(self.results.values())
        return self.results

    def display_results(self):
        """Prints the LCA results in a readable format."""
        print("\n--- Life Cycle Assessment Results ---")
        print(f"Impact Category: Global Warming Potential (GWP)\n")
        
        total = self.results.get('total_gwp', 0)
        if total == 0:
            print("Total GWP is zero. Cannot calculate percentages.")
            return

        for stage, gwp in self.results.items():
            if stage != 'total_gwp':
                percentage = (gwp / total) * 100 if total > 0 else 0
                print(f"{stage.replace('_', ' ').title():<15}: {gwp:12.2f} kg CO2eq ({percentage:.1f}%)")
        
        print("-" * 40)
        print(f"{'Total GWP':<15}: {total:12.2f} kg CO2eq (100.0%)")
        print("-------------------------------------\n")


if __name__ == '__main__':
    # --- Example Scenario: 1000 tons of Steel ---
    # These are placeholder values for a hypothetical scenario.
    steel_production_data = {
        # General
        "Metal_Type": "Steel",
        "Quantity_tons": 1000,
        # Extraction
        "Ore_Grade_%": 60,
        "Mining_Method": "Open-pit",
        "Explosives_Chemicals_kg": 5000,
        "Energy_Mining_MJ": 2000000,
        "Transport_Raw_km": 500,
        "Transport_Raw_Mode": "Train",
        # Manufacturing
        "Electricity_kWh": 600000,
        "Fuel_Coal_%": 70,
        "Fuel_Gas_%": 20,
        "Fuel_Renewables_%": 10,
        "Fuel_Hydro_%": 0,
        "Energy_Consumption_MJ": 15000000, # For furnaces, etc.
        "Emissions_CO_kg": 100,
        "Emissions_SO2_kg": 500,
        "Emissions_NOx_kg": 300,
        # Use Phase
        "Product_Lifetime_years": 50,
        "Energy_Use_MJ": 100, # Per year (e.g., for a building component with minimal energy use)
        # End of Life
        "Recycling_Rate_%": 85,
        "Landfilling_Rate_%": 15,
        "Transport_EOL_km": 150,
    }

    # 1. Create an instance of the calculator with the data
    lca_calculator = MetalLCA(**steel_production_data)

    # 2. Run the LCA calculation
    lca_calculator.run_lca()

    # 3. Display the formatted results
    lca_calculator.display_results()