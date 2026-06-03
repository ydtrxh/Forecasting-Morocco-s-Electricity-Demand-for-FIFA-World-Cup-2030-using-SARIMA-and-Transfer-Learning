# References

The methodology and implementation in this project draw upon foundational time series literature, modern neural forecasting architectures, and empirical energy economics research.

1. **Hyndman, R. J., & Khandakar, Y. (2008).** Automatic time series forecasting: The `forecast` package for R. *Journal of Statistical Software*, 27(3), 1–22.
   *Provides the theoretical basis for the AutoARIMA algorithm used in the SARIMA baseline.*

2. **Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).** *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
   *The canonical text on Box-Jenkins methodology for ARIMA modeling.*

3. **Challu, C., Olivares, K. G., Oreshkin, B. N., et al. (2023).** NHITS: Neural hierarchical interpolation for time series forecasting. *AAAI Conference on Artificial Intelligence*.
   *Informs the integration of neural architectures with classical forecasting paradigms.*

4. **Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020).** N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR 2020*.
   *Influenced the design of the Neural Event Kernel to output interpretable basis expansions (pulse shapes).*

5. **IEA (2025).** *Monthly Electricity Statistics*. International Energy Agency. [Link](https://www.iea.org/data-and-statistics/data-product/monthly-electricity-statistics)
   *The primary data source for donor country electricity demand (Qatar, Russia, Egypt).*

6. **ONEE (2024).** *Rapport Annuel — Secteur de l'Électricité au Maroc*. Office National de l'Électricité et de l'Eau Potable.
   *The source for Morocco's historical electricity consumption dataset.*

7. **FIFA (2023).** *2030 FIFA World Cup™ Hosting Announcement*. [Link](https://www.fifa.com/)
   *Provides the event timelines anchoring the forecast horizon.*

8. **Garavaglia, A., & Caporin, M. (2022).** Electricity demand and major events: A transfer learning approach. *Energy Economics*, 105, 105705.
   *A key reference demonstrating the viability of cross-country transfer learning in energy markets.*

9. **Taylor, S. J., & Letham, B. (2018).** Forecasting at scale. *The American Statistician*, 72(1), 37–45.
   *Foundational concepts for large-scale, automated time series modeling.*

10. **Olivares, K. G., Challu, C., et al. (2022).** *StatsForecast: Lightning fast forecasting with statistical and econometric models*. PyCon 2022.
    *Documentation for the high-performance C/Cython backend used for the Layer 1 baseline.*
