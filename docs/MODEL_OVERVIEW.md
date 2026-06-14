# Model overview

The code implements an exploratory CE-ABM with four main layers:

1. **Spatial layer**: a stylized Amsterdam grid initialized from neighborhood,
   CE-infrastructure and waste-burden data.
2. **Household layer**: representative household decision units with CE motives,
   barriers, recognition, household size and accumulated economic resources.
3. **Behavioral layer**: probabilistic sorting, container drop-off, bulky-waste
   drop-off and repair-cafe visit decisions.
4. **Outcome layer**: circularity indicators, unemployment under a reduced-form
   coupling, normalized spatial-advantage exposure inequality and accumulated
   economic-resource inequality.

The model is scenario-based. It compares different CE-support configurations
rather than forecasting exact Amsterdam trajectories.
