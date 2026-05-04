## Notes to mention in dataprocessing.ipynb

Weather features were generated from NASA POWER daily weather data and joined to the crop dataset using State_Name, District_Name, Crop_Year, Season and Crop.

Because not all crop observations had matching generated weather rows, the final benchmark dataset uses the weather-complete subset. The full joined dataset is kept as `data/crop_weather_joined.csv`.

The final weather-complete dataset covers years 1997-2014. A chronological split is used:

- train: 1997-2010
- validation: 2011-2012
- test: 2013-2014

Target is `yield`. `Production` is not used as a feature due to target leakage risk.
