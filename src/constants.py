"""Diagnostic ions, neutral losses, and mass spectrometry constants.

Sources:
  - Hsu & Turk (2003) J. Am. Soc. Mass Spectrom. 14:352-363
  - Brügger et al. (1997) PNAS 94:2339-2344
  - Murphy RC (2015) "Tandem Mass Spectrometry of Lipids" RSC
  - Castro-Perez et al. (2010) J. Proteome Res. 9:2377-2389
  - LIPID MAPS: https://www.lipidmaps.org
"""

import numpy as np

DIAGNOSTIC_IONS = {
    184.0733: "PC_choline_head",
    86.0964:  "PC_choline_frag",
    141.0191: "PE_head_NL",
    264.2686: "sphingosine_backbone",
    282.2792: "sphinganine_backbone",
    97.9769:  "phosphate_neg",
    79.9663:  "metaphosphate_neg",
    74.0368:  "glycerol_frag",
}

KNOWN_NEUTRAL_LOSSES = {
    'H2O': 18.0106, 'NH3': 17.0265, 'CO': 27.9949, 'CO2': 43.9898,
    'CH2': 14.0157, 'C2H4': 28.0313,
    'choline': 183.0660, 'ethanolamine': 141.0191,
    'serine': 185.0089, 'inositol': 260.0528,
    'phosphocholine': 183.0660, 'phosphate': 97.9769,
    'FA_C14:0': 228.2089, 'FA_C16:0': 256.2402, 'FA_C16:1': 254.2246,
    'FA_C18:0': 284.2715, 'FA_C18:1': 282.2559, 'FA_C18:2': 280.2402,
    'FA_C18:3': 278.2246, 'FA_C20:0': 312.3028, 'FA_C20:4': 304.2402,
    'FA_C22:6': 328.2402,
}

# 21 unique neutral loss masses used for edge features in spectral graph encoders
NEUTRAL_LOSS_MASSES = np.array([
    14.0157,   # CH2
    17.0265,   # NH3
    18.0106,   # H2O
    27.9949,   # CO
    28.0313,   # C2H4
    43.9898,   # CO2
    97.9769,   # phosphate
    141.0191,  # ethanolamine / PE_head_NL
    183.0660,  # choline / phosphocholine
    185.0089,  # serine
    228.2089,  # FA_C14:0
    254.2246,  # FA_C16:1
    256.2402,  # FA_C16:0
    260.0528,  # inositol
    278.2246,  # FA_C18:3
    280.2402,  # FA_C18:2
    282.2559,  # FA_C18:1
    284.2715,  # FA_C18:0
    304.2402,  # FA_C20:4
    312.3028,  # FA_C20:0
    328.2402,  # FA_C22:6
], dtype=np.float64)

NL_TOL = 0.02  # Da tolerance for neutral loss matching
