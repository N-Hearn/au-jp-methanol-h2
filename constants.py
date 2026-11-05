"""Physical constants and unit conversions used across all pathway modules.

These are fixed by chemistry and thermodynamics, not loaded from Excel.
"""

# Molar masses [g/mol]
M_H2   = 2.016  # https://webbook.nist.gov/cgi/cbook.cgi?ID=1333-74-0
M_CO2  = 44.010 # https://webbook.nist.gov/cgi/cbook.cgi?ID=124-38-9
M_H2O  = 18.015 # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185
M_MEOH = 32.042 # https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561

# Lower heating values [MJ/kg]
H2_LHV_MJ_PER_KG   = 120.21 # https://h2tools.org/hyarc/hydrogen-data/lower-and-higher-heating-values-hydrogen-and-other-fuels
MEOH_LHV_MJ_PER_KG = 19.9   # https://methanol.org/wp-content/uploads/2016/06/Physical-Properties-of-Pure-Methanol.pdf

# Time
HOURS_PER_YEAR = 8760.0
DAYS_PER_YEAR  = 365.0
HOURS_PER_DAY  = 24.0

# Unit conversions
KG_PER_TONNE  = 1000.0
KW_PER_MW     = 1000.0   # also kWh per MWh
L_PER_KL      = 1000.0
MJ_PER_GJ     = 1000.0
MTPA_TO_TONNE = 1e6

# Methanol cargo density at ambient tanker conditions [kg/m3]
MEOH_DENSITY_KG_PER_M3 = 793.0

# Stoichiometric mass ratios
MEOH_COMBUSTION_CO2_EF_KG_PER_KG = M_CO2 / M_MEOH          # CH3OH + 3/2 O2 -> CO2 + 2 H2O
CO2_STOICH_KG_PER_KG_MEOH        = M_CO2 / M_MEOH          # CH3OH + H2O -> CO2 + 3 H2 (cracking; same C-balance as combustion)
H2_STOICH_KG_PER_KG_MEOH         = 3.0 * M_H2 / M_MEOH     # CH3OH + H2O -> CO2 + 3 H2 (cracking)

# Fossil bunker fuel: IMO Cf for HFO (LCV 40,200 kJ/kg), MEPC.364(79) [t CO2/t fuel]
FUEL_OIL_CO2_EF_KG_PER_KG = 3.114
