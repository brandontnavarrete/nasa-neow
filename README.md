# The Great Space Race to Save the Earth
A Near Earth Object Classification project by Brandon Navarrete

# Goal
Humanity has forgotten how fragile we live. It would be wise to keep an eye to the stary abyss and determine if a near earth object has the potential to be hazardous.
With world climate becoming ever changing, we have lost sight on protecting the one resource we all share. EARTH.

* Here I will develop a model that can classify hazardous asteroids given their respective features of `diameter`, 'magnitude', and `velocity`

* This will be encompassed in a report that is easy to read and interpret to any viewers.


## Data Overview

* This data was pulled from kaggle(2023) which has been pulled from NASA's API

* 90836 rows, each it's own object or asteroids with 10 columns of its features

# Dependencies 

▪ [![python-shield](https://img.shields.io/badge/Python-864000?&logo=python&logoColor=white)
    ](https://www.python.org/)
▪ [![numpy-shield](https://img.shields.io/badge/Numpy-D44000?&logo=NumPy)
    ](https://numpy.org/)
▪ [![pandas-shield](https://img.shields.io/badge/Pandas-FF7A00?&logo=pandas)
    ](https://pandas.pydata.org/)

# Initial Questions
* How Many of our Objects Are Inert?

* Will Diameter play a Big Difference in Determining Hazard Status

* Will Relative Velocity play a Big Difference in Determining Hazard Status

* Will Absolute Magnitude play a Big Difference in Determining Hazard Status

# Data Dictionary
```
## :open_file_folder:   Data Dictionary
**Variable** |    **Value**    | **Meaning**
---|---|---
*ID* | numerical | Unique Identifier for each Asteroid
*Name* | string | Name given by NASA
*est_diameter_min* | Float | Minimum Estimated Diameter in Kilometeres.
*est_diameter_max* | Float | Maximum Estimated Diameter in Kilometeres.
*relative_velocity* | Float | Velocity Relative to Earth
*orbiting_body* | string | Earth
*sentry_object* | False | Included in sentry - an automated collision monitoring system
*absolute_magnitude* | Float | Describes intrinsic luminosity
*Hazardous* | Boolean |  Feature that shows whether asteroid is harmful or not
```

# Key Findings
* About 10 % of data was classified as `hazardous`
* All 3 features above shows promise in determing hazard status

# Recommendation
This model has a high percentage of finding the hazardous asteroids at the cost of a low accuracy, due to the false postitives

* This model should be used UNTIL a better model is developed

# Next Steps
* Use the API to gather more relevant features, try to increase hazardous object capture rate.

* Combine with image recogonition, try to automate process to have 24/7 observation / protection


# Steps To Clone:
1. Clone this repo
2. Import NASA's csv
3. Run Notebook
     # some dependencies may need to be installed such as 'xgboost'