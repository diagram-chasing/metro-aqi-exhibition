# BLR Metro PM2.5 Exhibition

Visualises what ambient PM2.5 across BLR would be if Namma Metro users drove instead of using transit.

## Existing Literature

1. [Subways and Urban Air Pollution](https://doi.org/10.3386/w24183)
   > For the set of cities in the top half of the initial AOD distribution, with above 0.36 AOD (approximately 28 μg/m3 pm2.5 ) on average in 2000, AOD levels fell by about 4% after the opening of the subway.

2. [The Effect of Metro Expansions on Air Pollution in Delhi](https://doi.org/10.1093/wber/lhv056)
   > Looking at the period 2004–2006, one of the larger rail extensions of the DM led to a 34 percent reduction in localized CO at a major traffic intersection in the city. Results for PM2.5 are also suggestive of a decline, while those for are inconclusive due to missing data. 

3. [Contribution of Subway Expansions to Air Quality Improvement and the Corresponding Health Implications in Nanjing, China](https://doi.org/10.3390/ijerph18030969)
   > The results reveal that subway expansions result in a statistically significant decrease in the air pollution level; specifically, the air pollution level experiences a 3.93% larger reduction in the areas close to subway lines.

4. [https://doi.org/10.1007/s43621-025-01994-0](https://doi.org/10.1007/s43621-025-01994-0)
   > The results show a net reduction of 155 tonnes of CO2 per day, along with daily savings of 3871 kg of CO, 2526 kg of HC, 133 kg of NOx, and 82 kg of PM. The greatest emission reductions were achieved by shifting vehicles from two-wheelers (12–48%), auto-rickshaws (19–47%), and taxis (5–34%), owing to their high emission intensity, while buses (9–14%) and cars (4–29%) contributed moderately.

## Modelling assumptions

Observed PM2.5 reflects the reduced emissions from current metro ridership. The net value adds back the emissions that metro riders avoid, giving the PM2.5 level that would exist in the absence of a metro network:

```
NetPM25[cell, hour] = ObsPM25[cell, hour] / (1 − ReductionFraction[cell])
```

### Reduction fraction

`ReductionFraction` is a static spatial map built from station ridership. Each station contributes an exponentially decaying influence weighted by its peak ridership:

```
score[cell] = Σ_s  ridership_s × exp(−distance(cell, s) / DECAY_KM)
```

The raw score is then normalised and floored so two calibration targets are met simultaneously:

| Target | Value | Basis |
|---|---|---|
| Peak `ReductionFraction` | 25% | Midpoint of 20–30% near-corridor reduction |
| City-wide mean `ReductionFraction` | 5% | ~4% AOD reduction in high-pollution cities |

```
ReductionFraction[cell] = clip(score_normalised[cell] + baseline, 0, 0.25)
```

| Parameter | Value | Role |
|---|---|---|
| `DECAY_KM` | 1.5 km | e-folding distance of each station's spatial influence |
| Peak normalisation | score / max(score) × 0.25 | Anchors highest-ridership cell to 25% |
| `baseline` | solved (~2–3%) | Uniform floor added so city mean = 5% |

Ridership is peak hourly entries + exits per station from the BMRCL parquet files. High ridership stations drive stronger local reductions; low ridership stations contribute proportionally less.

## Pipeline

| Script | Role |
|---|---|
| `hourly.py` | Fetches 24-hour PM2.5 readings from monitoring stations and interpolates to a 0.01° grid → `nc/hourly/gridded.nc` |
| `metro.py` | Builds the ridership-weighted `ReductionFraction` map → `nc/avoided/avoided_gridded.nc` |
| `main.py` | Adds back avoided emissions to observed PM2.5 to estimate PM2.5 without metro; saves NetCDF + CSVs + 24 hourly PNGs |

### Quick start

```bash
python hourly.py         # regenerate hourly PM2.5 grid
python metro.py          # regenerate reduction fraction map
python main.py           # compute net PM2.5 and render PNGs
```

Outputs are saved in `nc/`, `raw/`, and `png/`.
