# Raw tower data (not included)

The raw spreadsheets belong to the data providers of the Large-Scale Biosphere–Atmosphere Experiment in
Amazonia (LBA) and are not redistributed here. To rebuild `data/processed` from them, place the files below in
this folder (or pass another folder with `--raw`) and run `python reproduce.py data`.

| Site | Files expected by `scripts/01_prepare_data.py` |
|---|---|
| Rebio Jaru (2008–2010) | `Dados Rebio Jaru 2006 a 2008.xlsx` |
| Sinop (2001–2003) | `Apr01-Met.xls`, `May01-Met.xls`, `Cópia de Met-CO2 Jun-02.xls`, `Met Aug-03.xls`, `Met Sep-03.xls` (and, only for the QC log, `Met Nov-03.xls`, `Met Dec-03.xls`) |
| Cuieiras (2003–2004) | folder `Dados para Análises/` with `1.Variáveis Meteorológicas Cuieiras.xls`, `3.Irradiância Cuieiras.xls`, `4.Densidades de Fluxo de Energia Cuieiras.xls` |

All later stages start from `data/processed`, so the raw files are needed only to repeat the quality control.
